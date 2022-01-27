import os
import random
import time

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loading.label_adapters import scale_coords, xywh2xyxy
from utils.plots import plot_images
from loss.losses import ComputeLoss
from validation.abstract_validator import AbstractValidator
from logger import logger
from metrics.metrics import ConfusionMatrix, ap_per_class, box_iou, non_max_suppression
from utils.tta_utils import inference_with_tta


class YoloValidator(AbstractValidator):
    """YOLO model validator."""

    def __init__(
            self,
            model: nn.Module,  # type: ignore # noqa: F821
            dataloader: DataLoader,
            device: torch.device,
            cfg: Dict[str, Any],
            compute_loss: bool = True,
            log_dir: str = "exp",
            incremental_log_dir: bool = False,
            half: bool = False,
            export: bool = False,
            hybrid_label: bool = False,
            nms_type: str = "nms",
            tta: bool = False,
            tta_scales: List = None,
            tta_flips: List = None,
            show_plot: bool = False,
            num_plots2save: int = 0
    ) -> None:
        """Initialize YoloValidator class
        :param model: a torch model or TensorRT Wrapper
        :param dataloader: dataloader with validation dataset
        :param device: torch device
        :param cfg: validate config which includes
                    {
                        "train": {
                            "single_cls": True or False,
                            "plot": True or False,
                            "batch_size": number of batch size,
                            "image_size": image size
                        },
                        "hyper_params": {
                            "conf_t": confidence threshold,
                            "iou_t": IoU threshold
                        }
                    }
        :param log_dir: log directory path
        :param incremental_log_dir: use incremental directory
                                    If set, log_dir will be
                                        {log_dir}/val/{DATE}_runs,
                                        {log_dir}/val/{DATE}_runs1,
                                        {log_dir}/val/{DATE}_runs2,
                                                ...
        :param half: use half precision input
        :param export: export validation results to file.
        :param hybrid_label: Run NMS with hybrid information (ground truth label + predicted result.)
                            (PyTorch only) This is for auto-labeling purpose
        :param nms_type: NMS type (e.g. nms, batched_nms, fast_nms, matrix_nms)
        :param tta: Apply TTA or not
        :param tta_scales: scale ratios of each augmentation for TTA
        :param tta_flips: flip types of each augmentation for TTA
        :param show_plot: whether to show images with targets and detections
        :param num_plots2save: number of plots to save
        """
        super().__init__(
            model,
            dataloader,
            device,
            cfg,
            log_dir=log_dir,
            incremental_log_dir=incremental_log_dir,
            half=half,
            export=export,
            nms_type=nms_type,
            tta=tta,
            tta_scales=tta_scales,
            tta_flips=tta_flips,
        )
        self.class_map = list(range(self.n_class))  # type: ignore
        self.names = {k: v for k, v in enumerate(self.dataloader.dataset.names)}  # type: ignore
        self.confusion_matrix: Optional[ConfusionMatrix] = None
        self.statistics: Dict[str, Any] = {}
        self.nc = 1 if self.cfg_train["single_cls"] else int(self.n_class)  # type: ignore
        self.iouv = torch.linspace(0.5, 0.95, 10).to(self.device, non_blocking=True)  # IoU vector
        self.niou = self.iouv.numel()
        if compute_loss and not self.tta:
            self.loss_fn = ComputeLoss(self.model)
        else:
            self.loss_fn = None

        self.loss = torch.zeros(4, device=self.device)
        self.seen: int = 0
        self.hybrid_label = hybrid_label
        self.show_plots = show_plot
        num_batches = len(dataloader.dataset) // dataloader.batch_size  # noqa
        self._idxs2plot = random.sample(range(num_batches), min(num_plots2save, num_batches))
        self.tqdm = None  # type: Optional[tqdm]

    def init_statistics(self) -> None:
        """Set statistics default."""
        self.statistics = {
            "dt": [0.0, 0.0, 0.0],
            "p": 0.0,
            "r": 0.0,
            "f1": 0.0,
            "f2": 0.0,
            "mp": 0.0,
            "mr": 0.0,
            "mf1": 0.0,
            "mf2": 0.0,
            "map50": 0.0,
            "map": 0.0,
            "jdict": [],
            "stats": [],
            "ap": [],
            "ap_class": [],
        }

    def init_attrs(self) -> None:
        """Initialize attributes before validation."""
        self.confusion_matrix = ConfusionMatrix(nc=self.n_class)
        self.init_statistics()
        self.seen = 0
        self.tqdm = tqdm(
            enumerate(self.dataloader),
            desc="Validating ...",
            total=len(self.dataloader),
        )

    def prepare_img(self, img: torch.Tensor) -> torch.Tensor:
        """Prepare img for model."""
        img = img.to(self.device, non_blocking=True)
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        return img

    def convert_trt_out(self, out: torch.Tensor, n_objs: torch.Tensor) -> List[torch.Tensor]:
        """Convert output from TensorRT model to validation ready format
        :param out: (batch_size, keep_top_k, 6) tensor
        :param n_objs: (batch_size,) tensor which contains detected objects on each image
        :return: List of detected results (n_obj, 6) (x1, y1, x2, y2, confidence, class_id)
        """
        result = [
            torch.zeros((n_obj, 6)).to(self.device, non_blocking=True)
            for n_obj in n_objs
        ]

        for i, n_obj in enumerate(n_objs):
            result[i] = out[i][:n_obj]

        return result

    def compute_loss(self, train_out: torch.Tensor, targets: torch.Tensor) -> None:
        """Compute loss.
        :param train_out: output from model (detected)
        :param targets: target labels
        """
        self.loss += self.loss_fn([x.float() for x in train_out], targets)[1][:4]

    def process_batch(
            self,
            detections: Union[torch.Tensor, np.ndarray],
            labels: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """Return correct predictions matrix
        Both sets of boxes are in (x1, y1, x2, y2) format
        :param detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        :param labels (Array[M, 5]), class, x1, y1, x2, y2
        :returns: correct (Array[N, 10]), for 10 IoU levels
        """
        correct = torch.zeros(
            detections.shape[0],
            self.iouv.shape[0],
            dtype=torch.bool,
            device=self.iouv.device,
        )
        iou = box_iou(labels[:, 1:], detections[:, :4])
        x = torch.where(
            (iou >= self.iouv[0]) & (labels[:, 0:1] == detections[:, 5])
        )  # IoU above threshold and classes match
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            )  # [label, detection, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            matches = torch.Tensor(matches).to(self.iouv.device, non_blocking=True)
            correct[matches[:, 1].long()] = matches[:, 2:3] >= self.iouv
        return correct

    def statistics_per_image(
            self,
            img: torch.Tensor,
            out: list,
            targets: torch.Tensor,
            shapes: tuple,
            paths: tuple,
    ) -> None:
        """Calculate statistics per image
        :param img: input image
        :param out: model output of input image (img)
        :param targets: target label
        :param shapes: batch image shape
        :param paths: image path
        """
        for si, pred in enumerate(out):
            if si >= len(shapes):  # TensorRT works with fixed batch size only
                break

            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]  # noqa
            self.seen += 1

            if len(pred) == 0:
                if nl:
                    self.statistics["stats"].append(
                        (
                            torch.zeros(0, self.niou, dtype=torch.bool),
                            torch.Tensor(),
                            torch.Tensor(),
                            tcls,
                        )
                    )
                continue

            # Predictions
            if self.cfg_train["single_cls"]:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(
                img[si].shape[1:], predn[:, :4], shape, shapes[si][1]
            )  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(
                    img[si].shape[1:], tbox, shape, shapes[si][1]
                )  # native-space labels
                labels_nat = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = self.process_batch(predn, labels_nat)
                if self.cfg_train["plot"]:
                    self.confusion_matrix.process_batch(predn, labels_nat)
            else:
                correct = torch.zeros(pred.shape[0], self.niou, dtype=torch.bool)
            self.statistics["stats"].append(
                (correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls)
            )  # (correct, conf, pcls, tcls)

    def validation_step(
            self,
            val_batch: Tuple[
                torch.Tensor,
                torch.Tensor,
                Tuple[str, ...],
                Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...],
            ],
            batch_idx: int,
    ) -> None:
        """Validate a step
        :param val_batch: a validation batch
        :param batch_idx: batch index
        """
        imgs, targets, paths, shapes = val_batch
        t1 = time.time()
        imgs = self.prepare_img(imgs)
        targets = targets.to(self.device, non_blocking=True)
        batch_size, _, height, width = imgs.shape
        t2 = time.time()
        self.statistics["dt"][0] += t2 - t1

        # Run model
        if self.tta:
            outs = inference_with_tta(
                self.model,
                imgs.half() if self.half else imgs,
                self.tta_scales,
                self.tta_flips,
            )
        else:
            outs = self.model(
                imgs.half() if self.half else imgs
            )  # inference and training outputs
        self.statistics["dt"][1] += time.time() - t2

        if len(outs) == 2:
            out, train_out = outs
        else:
            out, train_out = outs[0], None

        targets = self.convert_target(targets, width, height)
        labels_for_hybrid = (
            [targets[targets[:, 0] == i, 1:] for i in range(batch_size)]
            if self.hybrid_label
            else []
        )

        # Compute loss
        if self.loss_fn:
            self.compute_loss(train_out, targets)

        t3 = time.time()
        if isinstance(train_out, torch.Tensor):  # TensorRT case.
            out = self.convert_trt_out(out.clone(), train_out.clone())
        else:
            out = non_max_suppression(
                out,
                self.cfg_hyp["conf_t"],
                self.cfg_hyp["iou_t"],
                multi_label=True,
                labels=labels_for_hybrid,
                agnostic=self.cfg_train["single_cls"],
                nms_type=self.nms_type,
            )
        self.statistics["dt"][2] += time.time() - t3
        self.statistics_per_image(imgs, out, targets, shapes, paths)

        if len(self._idxs2plot) > 0 or self.show_plots:
            formatted_out = \
                [
                    np.stack(
                        [np.array(
                            [float(i), c, (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2, (y2 - y1) / 2])
                            for (x1, y1, x2, y2, _, c) in arr.cpu().numpy()])
                    if len(arr) > 0 else np.array([])
                    for i, arr in enumerate(out)
                ]
            f_name = None
            if batch_idx in self._idxs2plot:
                f_name = os.path.join(self.log_dir, f"img_batch{batch_idx}.jpg")
            out_img = plot_images(images=imgs, targets=targets, predictions=formatted_out,
                                  unnormalize=False, fname=f_name, resize_mosaic=False)
            if self.show_plots:
                plt.imshow(out_img)
                plt.show()

    def compute_statistics(self) -> None:
        """Compute statistics for dataset."""
        self.statistics["stats"] = [
            np.concatenate(x, 0) for x in zip(*self.statistics["stats"])
        ]  # to numpy
        if len(self.statistics["stats"]) and self.statistics["stats"][0].any():
            (
                self.statistics["p"],
                self.statistics["r"],
                self.statistics["ap"],
                self.statistics["f1"],
                self.statistics["f2"],
                self.statistics["ap_class"],
            ) = ap_per_class(
                *self.statistics["stats"],
                plot=self.export,
                save_dir=self.log_dir,
                names=list(self.names.keys()),
            )
            self.statistics["ap50"], self.statistics["ap"] = (
                self.statistics["ap"][:, 0],
                self.statistics["ap"].mean(1),
            )  # AP@0.5, AP@0.5:0.95
            (
                self.statistics["mp"],
                self.statistics["mr"],
                self.statistics["map50"],
                self.statistics["map"],
                self.statistics["mf1"],
                self.statistics["mf2"],
            ) = (
                self.statistics["p"].mean(),
                self.statistics["r"].mean(),
                self.statistics["ap50"].mean(),
                self.statistics["ap"].mean(),
                self.statistics["f1"].mean(),
                self.statistics["f2"].mean(),
            )
        self.statistics["nt"] = \
            np.bincount(
                self.statistics["stats"][3].astype(np.int64), minlength=self.nc
            )  # number of targets per class

    def print_results(self, verbose: bool = False) -> tuple:
        """Print validation results.
        :param verbose: print validation result per class or not.
        :returns: a tuple with dt for statistics.
        """
        # print result
        s = ("%20s" + "%11s" * 8) % (
            "Class",
            "Images",
            "Labels",
            "P",
            "R",
            "F1",
            "F2",
            "mAP@.5",
            "mAP@.5:.95",
        )

        pf = "%20s" + "%11i" * 2 + "%11.3g" * 6  # print format
        log_str = str(
            pf
            % (
                "all",
                self.seen,
                self.statistics["nt"].sum(),
                self.statistics["mp"],
                self.statistics["mr"],
                self.statistics["mf1"],
                self.statistics["mf2"],
                self.statistics["map50"],
                self.statistics["map"],
            )
        )

        logger.info("\n" + s + "\n" + log_str)

        # print result per class
        if (verbose or self.nc < 50) and self.nc > 1 and len(self.statistics["stats"]):
            for i, c in enumerate(self.statistics["ap_class"]):
                logger.info(
                    str(
                        pf
                        % (
                            self.names[c],
                            self.seen,
                            self.statistics["nt"][c],
                            self.statistics["p"][i],
                            self.statistics["r"][i],
                            self.statistics["f1"][i],
                            self.statistics["f2"][i],
                            self.statistics["ap50"][i],
                            self.statistics["ap"][i],
                        )
                    )
                )
        # print speed
        t = tuple(
            x / self.seen * 1e3 for x in self.statistics["dt"]
        )  # speeds per image
        if verbose:
            shape = (
                self.cfg_train["batch_size"],
                3,
                self.cfg_train["image_size"],
                self.cfg_train["image_size"],
            )
            logger.info(
                f"Speed: {t[0]:.1f}ms pre-process, {t[1]:.1f}ms inference, {t[2]:.1f}ms NMS per image at shape {shape}"
            )
        return t

    @torch.no_grad()
    def validation(self, verbose: bool = False) -> Tuple[Tuple[list, ...], np.ndarray, tuple]:  # type: ignore
        """Validate model
        :return metrics: mean precision, recall, F1-score, F2-score, MAP50, MAP, losses, maps and load speed
        """

        self.init_attrs()
        # dt, precision, recall, f1 score, mean-precision, mean-recall, mAP@.5, mAP@.5:.95

        for batch_i, batch in self.tqdm:
            self.validation_step(batch, batch_i)
        self.compute_statistics()

        t = self.print_results(verbose=verbose)

        maps = np.zeros(self.nc) + self.statistics["map"]
        for i, c in enumerate(self.statistics["ap_class"]):
            maps[c] = self.statistics["ap50"][i]

        return (
            (
                self.statistics["mp"],
                self.statistics["mr"],
                self.statistics["mf1"],
                self.statistics["mf2"],
                self.statistics["map50"],
                self.statistics["map"],
                *(self.loss.cpu() / len(self.dataloader)).tolist(),
            ),
            maps,
            t,
        )
