import math
import os
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as functional
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loss.losses import ComputeLoss
from training.abstract_trainer import AbstractTrainer
from utils.general import check_img_size, labels_to_image_weights
from logger import colorstr, logger
from utils.plots import plot_images, plot_label_histogram
from training.early_stopper import EarlyStopping
from utils.torch_utils import de_parallel
from validation.validator import YoloValidator

from utils.torch_utils import ModelEMA

LOCAL_RANK = int(
    os.getenv("LOCAL_RANK", -1)
)  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


class Trainer(AbstractTrainer):
    """Trainer class."""

    def __init__(
            self,
            model: nn.Module,
            cfg: Dict[str, Any],
            train_dataloader: DataLoader,
            val_dataloader: Optional[DataLoader],
            ema: Optional["ModelEMA"],
            device: torch.device,
            log_dir: str = "exp",
            tensorboard: bool = True,
            incremental_log_dir: bool = False,
            use_swa: bool = False,
    ) -> None:
        """Initialize YoloTrainer class
        :param model: yolo model to train
        :param cfg: config
        :param train_dataloader: dataloader for training
        :param val_dataloader: dataloader for validation
        :param tensorboard: whether to save metrics in a tensorboard file
        :param use_swa: apply SWA (Stochastic Weight Averaging) or not
        """
        super().__init__(
            model,
            cfg,
            train_dataloader,
            val_dataloader,
            device=device,
            log_dir=log_dir,
            incremental_log_dir=incremental_log_dir,
            use_swa=use_swa,
        )

        self.cfg_hyp["label_smoothing"] = self.cfg_train["label_smoothing"]
        self.ema = ema
        self.best_score = 0.0

        self.loss = ComputeLoss(self.model)
        self.nbs = 64  # nominal batch size
        self.accumulate = max(round(self.nbs / self.cfg_train["batch_size"]), 1)
        self.pretrained_optimizer = False
        self.optimizer, self.scheduler = self._init_optimizer()
        self.val_maps = np.zeros(self.model.nc)  # map per class
        self.scaler = None  # type: Optional[amp.GradScaler]
        self.mloss = None  # type Optional[torch.Tensor]
        self.num_warmups = max(
            round(self.cfg_hyp["warmup_epochs"] * len(self.train_dataloader)), 1_000
        )
        if isinstance(self.cfg_train["image_size"], int):
            self.cfg_train["image_size"] = [self.cfg_train["image_size"]] * 2
        self.img_size, self.val_img_size = [
            # check_img_size(x, max(self.model.stride))
            check_img_size(x, self.model.stride)
            for x in self.cfg_train["image_size"]
        ]
        self.cfg_train["world_size"] = (
            int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        )

        self.tb_writer = SummaryWriter(log_dir) if tensorboard else None
        if self.val_dataloader is not None:
            self.validator = YoloValidator(
                self.model if self.ema is None else self.ema.ema,
                self.val_dataloader,
                self.device,
                cfg,
                log_dir=self.log_dir,
            )
        patience = self.cfg_train["patience"]
        self.stopper = EarlyStopping(patience, self.best_score)

    def _lr_function(self, epoch: float) -> float:
        """
        Computes a multiplicative factor given an integer parameter epoch to adjust the learning rate
        :param epoch: current epoch
        :return: factor according to the current epoch
        """
        if "linear_lr" in self.cfg_train.keys() and self.cfg_train["linear_lr"]:
            return (1 - epoch / (self.cfg_train["epochs"] - 1)) * (
                    1.0 - self.cfg_hyp["lrf"]
            ) + self.cfg_hyp["lrf"]

        return ((1 + math.cos(epoch * math.pi / self.cfg_train["epochs"])) / 2) * (
                1 - self.cfg_hyp["lrf"]
        ) + self.cfg_hyp["lrf"]

    def _init_optimizer(self) -> Tuple[List[optim.Optimizer], List[lr_scheduler.LambdaLR]]:
        """Initialize optimizer and scheduler."""
        self.cfg_hyp["weight_decay"] *= self.cfg_train["batch_size"] * self.accumulate / self.nbs
        logger.info(f"Scaled weight_decay = {self.cfg_hyp['weight_decay']}")

        pg0: List[torch.Tensor] = []
        pg1: List[torch.Tensor] = []
        pg2: List[torch.Tensor] = []

        for _, v in self.model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, torch.Tensor):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, torch.Tensor):
                pg1.append(v.weight)

        optimizer = getattr(
            __import__("torch.optim", fromlist=[""]), self.cfg_hyp["optimizer"]
        )(params=pg0, **self.cfg_hyp["optimizer_params"])

        optimizer.add_param_group(
            {"params": pg1, "weight_decay": self.cfg_hyp["weight_decay"]}
        )
        optimizer.add_param_group({"params": pg2})
        logger.info(
            "Optimizer groups: %g .bias, %g conv.weight, %g other"
            % (len(pg2), len(pg1), len(pg0))
        )
        logger.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
            f"{len(pg0)} weight, {len(pg1)} weight (no decay), {len(pg2)} bias"
        )

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self._lr_function)

        pretrained = self.cfg_train.get("weights", "").endswith(".pt")
        if pretrained:
            ckpt = torch.load(self.cfg_train["weights"])
            if ckpt.get("optimizer") is not None:
                self.pretrained_optimizer = True
                optimizer.load_state_dict(ckpt["optimizer"][0])
                self.best_score = ckpt["best_score"]

            if self.ema and ckpt.get("ema"):
                self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
                self.ema.updates = ckpt["updates"]

        return [optimizer], [scheduler]

    def warmup(self, ni: int, epoch: int) -> None:
        """Warmup before training
        :param ni: number integrated batches
        :param epoch: current epoch
        """
        x_intp = [0, self.num_warmups]
        self.accumulate = max(
            1,
            np.interp(ni, x_intp, [1, self.nbs / self.cfg_train["batch_size"]]).round(),
        )
        for optimizer in self.optimizer:
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x["lr"] = np.interp(
                    ni,
                    x_intp,
                    [
                        self.cfg_hyp["warmup_bias_lr"] if j == 2 else 0.0,
                        x["initial_lr"] * self._lr_function(epoch),
                    ],
                )
                if "momentum" in x:
                    x["momentum"] = np.interp(
                        ni,
                        x_intp,
                        [self.cfg_hyp["warmup_momentum"], self.cfg_hyp["momentum"]],
                    )

    def multi_scale(self, imgs: torch.Tensor) -> torch.Tensor:
        """Set for multi scale image training
        :param imgs: torch tensor images
        :returns: Reshaped images with scale factor
        """
        grid_size = self.model.stride
        sz = (
                random.randrange(self.img_size * 0.5, self.img_size * 1.5 + grid_size)
                // grid_size
                * grid_size
        )

        scale_factor = sz / max(imgs.shape[2:])
        if scale_factor != 1:
            new_shape = [
                math.ceil(x * scale_factor / grid_size) * grid_size
                for x in imgs.shape[2:]
            ]
            imgs = functional.interpolate(
                imgs, size=new_shape, mode="bilinear", align_corners=False
            )
        return imgs

    def print_intermediate_results(
            self,
            loss_items: torch.Tensor,
            t_shape: torch.Size,
            img_shape: torch.Size,
            epoch: int,
            batch_idx: int,
    ) -> str:
        """Print intermediate_results during training batches
        :param loss_items: loss items from model
        :param t_shape: torch label shape
        :param img_shape: torch image shape
        :param epoch: current epoch
        :param batch_idx: current batch index
        :returns: string for print
        """
        self.mloss = (self.mloss * batch_idx + loss_items) / (batch_idx + 1)
        mem = "%.3gG" % (
            torch.cuda.memory_reserved() / 1e9  # to GBs
            if torch.cuda.is_available()
            else 0
        )
        str2show = ("%10s" * 2 + "%10.4g" * 6) % (
            "%g/%g" % (epoch, self.epochs - 1),
            mem,
            *self.mloss,
            t_shape[0],
            img_shape[-1],
        )

        if self.pbar:
            self.pbar.set_description(str2show)
        return str2show

    def training_step(
            self,
            train_batch: Tuple[
                torch.Tensor,
                torch.Tensor,
                Tuple[str, ...],
                Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...],
            ],
            batch_idx: int,
            epoch: int,
    ) -> torch.Tensor:
        """Train a step.
        :param train_batch: batch data
        :param batch_idx: batch index
        :param epoch: current epoch
        :returns: Result of loss function
        """
        num_integrated_batches = batch_idx + len(self.train_dataloader) * epoch

        if not self.pretrained_optimizer and num_integrated_batches <= self.num_warmups:
            self.warmup(num_integrated_batches, epoch)

        imgs, labels, paths, shapes = train_batch
        imgs = self.prepare_img(imgs)
        labels = labels.to(self.device)

        if self.cfg_train["multi_scale"]:
            imgs = self.multi_scale(imgs)

        with amp.autocast(enabled=self.cuda):
            pred = self.model(imgs)
            loss, loss_items = self.loss(pred, labels)
            if RANK != -1:
                loss *= self.cfg_train["world_size"]

        # backward
        self.scaler.scale(loss).backward()

        # Optimize
        if num_integrated_batches % self.accumulate == 0:
            for optimizer in self.optimizer:
                self.scaler.step(optimizer)  # optimizer.step
                self.scaler.update()
                optimizer.zero_grad()
                if self.ema:
                    self.ema.update(self.model)

        if RANK in [-1, 0]:
            self.print_intermediate_results(loss_items, labels.shape, imgs.shape, epoch, batch_idx)

            if num_integrated_batches < 3:
                # plot images.
                f_name = os.path.join(self.log_dir, f"train_batch{num_integrated_batches}.jpg")
                plot_images(images=imgs, targets=labels, paths=paths, fname=f_name)

        self.log_dict({"step_loss": loss[0].item()})

        return loss[0]

    def log_dict(self, data: Dict[str, Any]) -> None:
        """Log dictionary data."""
        super().log_dict(data)
        self.update_loss()

    def update_loss(self) -> None:
        """Update train loss by `step_loss`."""
        if not self.state["is_train"]:
            return
        train_log = self.state["train_log"]
        if "loss" not in train_log:
            train_log["loss"] = 0
        train_log["loss"] += train_log["step_loss"]

    def _save_weights(self, epoch: int, w_name: str) -> None:
        if RANK in [-1, 0]:
            ckpt = {
                "epoch": epoch,
                "best_score": self.best_score,
                "model": deepcopy(de_parallel(self.model)).half(),
                "optimizer": [optimizer.state_dict() for optimizer in self.optimizer],
            }
            if self.use_swa:
                ckpt["mAP50"] = self.state["val_log"]["mAP50"]
            if self.ema is not None:
                ckpt.update(
                    {"ema": deepcopy(self.ema.ema).half(), "updates": self.ema.updates}
                )

            torch.save(ckpt, os.path.join(self.wdir, w_name))
            del ckpt

    def validation(self) -> None:
        """Validate model."""
        if RANK in [-1, 0]:
            val_result = self.validator.validation()
            metrics_dict = {
                "mP": val_result[0][0],
                "mR": val_result[0][1],
                "mF1": val_result[0][2],
                "mF2": val_result[0][3],
                "mAP50": val_result[0][4],
                "mAP50_95": val_result[0][5],
                "box_loss": val_result[0][6],
                "obj_loss": val_result[0][7],
                "cls_loss": val_result[0][8],
                "total_loss": val_result[0][9],
                "mAP50_by_cls": {
                    k: val_result[1][i]
                    for i, k in enumerate(self.val_dataloader.dataset.names)
                },
            }  # type:Dict[str, Union[float, dict]]
            self.log_dict(metrics_dict)

            # Write results with tensorboard writer if one exists
            if self.tb_writer:
                # Store metrics
                for label, val in metrics_dict.items():
                    if isinstance(val, float) or isinstance(val, int):
                        if "loss" in label:
                            self.tb_writer.add_scalar("val/" + label, val, self.current_epoch)
                        else:
                            self.tb_writer.add_scalar("metrics/" + label, val, self.current_epoch)

                # Store loss values
                loss_labels = ["box_loss", "obj_loss", "cls_loss", "total_loss"]
                for label, val in zip(loss_labels, self.mloss):
                    self.tb_writer.add_scalar("train/" + label, val.item(), self.current_epoch)

                # self.tb_writer.add_scalar("val/loss", self.validator.loss)
            self.val_maps = val_result[1]

            if metrics_dict["mF2"] > self.best_score:
                self.best_score = metrics_dict["mF2"]

            self._save_weights(self.current_epoch, "last.pt")

            if metrics_dict["mF2"] == self.best_score:
                self.best_score = metrics_dict["mF2"]
                self._save_weights(self.current_epoch, "best.pt")

            if RANK == -1 and self.stopper(epoch=self.current_epoch, score=metrics_dict["mF2"]):
                self.is_early_stop = True

    def update_image_weights(self) -> None:
        """Update image weights."""
        if self.cfg_train["image_weights"]:
            n_imgs = 0
            # Generate indices
            if RANK in [-1, 0]:
                # number of total images
                n_imgs = len(self.train_dataloader.dataset.img_files)

                # class weights
                class_weights = (
                        self.model.class_weights.cpu().numpy() * (1 - self.val_maps) ** 2
                )
                # images weights
                image_weights = labels_to_image_weights(
                    self.train_dataloader.dataset.labels,
                    nc=self.model.nc,
                    class_weights=class_weights,
                )

                self.train_dataloader.dataset.indices = random.choices(
                    range(n_imgs), weights=image_weights, k=n_imgs
                )

            # Broadcast if DDP
            if RANK != -1:
                indices = (
                    torch.tensor(self.train_dataloader.dataset.indices)
                    if RANK == 0
                    else torch.zeros(n_imgs)
                ).int()
                dist.broadcast(indices, 0)
                if RANK != 0:
                    self.train_dataloader.dataset.indices = indices.cpu().numpy()

    def set_datasampler(self, epoch: int) -> None:
        """Set dataloader's sampler epoch"""
        # if RANK != -1:
        #     self.train_dataloader.sampler.set_epoch(epoch)
        pass

    def on_epoch_start(self, epoch: int) -> None:
        """Run on an epoch starts
        :param epoch: current epoch
        """
        self.update_image_weights()
        self.set_datasampler(epoch)
        self.log_train_stats()
        self.set_trainloader_tqdm()
        self.mloss = torch.zeros(4, device=self.device)
        for optimizer in self.optimizer:
            optimizer.zero_grad()

    def on_epoch_end(self, epoch: int) -> None:
        """Run on an epoch ends
        :param epoch: current epoch
        """
        for optimizer in self.optimizer:  # for tensorboard
            lr = [x["lr"] for x in optimizer.param_groups]  # noqa

        self.scheduler_step()
        if RANK in [-1, 0] and self.ema is not None:
            self.update_ema_attr()
        # average the cumulated loss
        self.state["train_log"]["loss"] /= len(self.train_dataloader.dataset)  # noqa: dataset has __len__

    def on_validation_end(self) -> None:
        """Run on validation end"""
        if self.state["val_log"]:
            self.state["val_log"].pop("mAP50_by_cls", None)

    def scheduler_step(self) -> None:
        """Update scheduler parameters"""
        for scheduler in self.scheduler:
            scheduler.step()

    def update_ema_attr(self, include: Optional[List[str]] = None) -> None:
        """Update ema attributes
        :param include: a list of string which contains attributes
        """
        if not include:
            include = ["yaml", "nc", "hyp", "gr", "names", "stride", "cfg"]
        if self.ema:
            self.ema.update_attr(self.model, include=include)

    def set_grad_scaler(self) -> amp.GradScaler:
        """Set GradScaler"""
        return amp.GradScaler(enabled=self.cuda)

    def set_trainloader_tqdm(self) -> None:
        """Set tqdm object of train dataloader"""
        if RANK in [-1, 0]:
            self.pbar = tqdm(
                enumerate(self.train_dataloader), total=len(self.train_dataloader)
            )

    def log_train_stats(self) -> None:
        """Log train information table headers"""
        if RANK in [-1, 0]:
            logger.info(
                f"Epoch {self.current_epoch}:" +
                ("\n" + "%10s" * 8)
                % (
                    "Epoch",
                    "gpu_mem",
                    "box",
                    "obj",
                    "cls",
                    "total",
                    "targets",
                    "img_size",
                )
            )

    def on_train_start(self) -> None:
        """Run on start training"""
        labels = np.concatenate(self.train_dataloader.dataset.labels, 0)
        mlc = labels[:, 0].max()  # type: ignore
        nc = len(self.train_dataloader.dataset.names)
        assert mlc < nc, f"Label class {mlc} exceeds nc={nc}. Possible class labels are 0-{nc - 1}"

        grid_size = self.model.stride  # type: ignore
        imgsz, _ = [check_img_size(x, grid_size) for x in self.cfg_train["image_size"]]

        if RANK in [-1, 0] and not self.cfg_train["resume"]:
            plot_label_histogram(labels, save_dir=self.log_dir)

        for scheduler in self.scheduler:
            scheduler.last_epoch = self.start_epoch - 1  # type: ignore

        self.scaler = self.set_grad_scaler()

    def on_train_end(self) -> None:
        """Run on the end of the training."""
        self._save_weights(-1, "last.pt")

    def log_train_cfg(self) -> None:
        """Log train configurations."""
        if RANK in [-1, 0]:
            logger.info(
                "Image sizes %g train, %g test\n"
                "Using %g dataloader workers\nLogging results to %s\n"
                "Starting training for %g epochs..."
                % (
                    self.cfg_train["image_size"][0],
                    self.cfg_train["image_size"][0],
                    self.train_dataloader.num_workers,
                    self.log_dir,
                    self.epochs,
                )
            )
