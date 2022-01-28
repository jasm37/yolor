import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import torch
from torch import nn

from utils.torch_utils import ModelEMA, load_model_weights
from utils.general import check_img_size, labels_to_class_weights
from logger import logger, colorstr
from utils.torch_utils import is_parallel

LOCAL_RANK = int(
    os.getenv("LOCAL_RANK", -1)
)  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


class AbstractModelManager(ABC):
    """Abstract model manager"""

    def __init__(
            self,
            model: nn.Module,
            cfg: Dict[str, Any],
            device: torch.device,
            weight_dir: Path,
    ) -> None:
        """Initialize model manager
        :param model: PyTorch model
        :param cfg: training config
        :param device: device to load the model weight
        :param weight_dir: weight directory path
        """
        self.model = model
        self.cfg = cfg
        self.device = device
        self.weight_dir = weight_dir
        self.start_epoch = 0

        if hasattr(self.model, "model_parser"):
            self.yaml = self.model.model_parser.cfg  # type: ignore
        else:
            self.yaml = None

    @abstractmethod
    def _load_weight(self, path: str) -> nn.Module:
        """Abstract Load model weight
        Read weights from the path and load them on to the model
        :return: weights loaded model
        """
        pass


class YOLOModelManager(AbstractModelManager):
    """YOLO Model Manager."""

    def __init__(
            self,
            model: nn.Module,
            cfg: Dict[str, Any],
            device: torch.device,
            weight_dir: Path,
    ) -> None:
        """Initialize YOLO Model manager
        :param model: PyTorch model
        :param cfg: training config
        :param device: device to load the model weight
        :param weight_dir: weight directory path
        """
        super().__init__(model, cfg, device, weight_dir)

    def _load_weight(self, path: str) -> nn.Module:
        """Load weights from the model.
        Also, reads parameters to decide whether the model
        training has been completed or needs to be resumed
        :return: weights loaded model
        """
        if path.endswith(".pt"):
            ckpt = torch.load(path, map_location=self.device)
            exclude: List[str] = []
            self.model = load_model_weights(self.model, weights=ckpt, exclude=exclude)
            start_epoch = ckpt["epoch"] + 1
            if self.cfg["train"]["resume"]:
                assert start_epoch > 0, (
                        "%s training to %g epochs is finished, nothing to resume."
                        % (self.cfg["train"]["weights"], self.cfg["train"]["epochs"],)
                )
                if RANK in [-1, 0]:
                    logger.info(
                        "Copying "
                        + colorstr("bold", f"{Path(path).parent.parent}")
                        + " to "
                        + colorstr("bold", f"{self.weight_dir.parent} ...")
                    )
                    shutil.copytree(
                        Path(path).parent.parent,
                        self.weight_dir.parent,
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.yaml"),
                    )  # save previous files
                    shutil.copytree(
                        Path(path).parent.parent,
                        self.weight_dir.parent / f"backup_epoch{start_epoch - 1}",
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.pt"),
                    )  # save previous weights
                    shutil.copytree(
                        Path(path).parent,
                        self.weight_dir.parent / f"weight_epoch{start_epoch - 1}",
                        dirs_exist_ok=True,
                    )  # save previous weights
                self.start_epoch = start_epoch
            if self.cfg["train"]["epochs"] < start_epoch:
                logger.info(
                    "%s has been trained for %g epochs. Fine-tuning for %g additional epochs."
                    % (
                        self.cfg["train"]["weights"],
                        ckpt["epoch"],
                        self.cfg["train"]["epochs"],
                    )
                )
                self.start_epoch = 0

        return self.model

    def freeze(self, freeze_n_layer: int) -> nn.Module:
        """Freeze layers from the top.
        :param freeze_n_layer: freeze from the top to nth layer.
                              0 will set all parameters to be trainable.
                              i.e. freeze_n_layer = 3 will freeze
                              model.0.*
                              model.1.*
                              model.2.*
        :return: frozen model.
        """
        freeze_list = [f"model.{x}." for x in range(freeze_n_layer)]
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze_list):
                logger.info(f"freezing {k}")
                v.requires_grad = False

        return self.model

    def set_model_params(
            self,
            class_names: List[str],
            labels: List[np.ndarray],
            ema: Optional["ModelEMA"] = None  # noqa
    ) -> nn.Module:
        """Set necessary model parameters required in YOLO.
        :param class_names: data set class names
        :param labels: data set labels to generate class weights
        :param ema: ema model to be set
        :return: self.model with parameters
        """
        head = self.model.module_list[-1]
        models = [self.model]
        grid_size = head.stride

        image_size = check_img_size(self.cfg["train"]["image_size"], grid_size)

        if ema:
            models.append(ema.ema)

        if is_parallel(self.model):
            models.append(self.model.module)  # type: ignore

        for model in models:
            model.nc = len(class_names)  # type: ignore
            model.hyp = self.cfg["hyper_params"]
            model.gr = 1.0  # type: ignore
            model.class_weights = labels_to_class_weights(labels, len(class_names)).to(self.device)

            model.names = class_names  # type: ignore
            model.stride = head.stride  # type: ignore
            model.cfg = self.cfg  # type: ignore
            model.yaml = self.yaml  # type: ignore

            # Update loss weight hyper params
            # scale box loss with the number of head
            model.hyp["cls"] *= (  # type: ignore
                    model.nc / 80.0  # type: ignore
            )  # scale to classes and layers
            model.hyp["obj"] *= (  # type: ignore
                    (image_size / 640) ** 2
            )  # scale box loss with image size

        return self.model
