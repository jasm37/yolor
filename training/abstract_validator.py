import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from logger import logger, colorstr
from utils.general import increment_path


class AbstractValidator(ABC):
    """Model validator class."""

    def __init__(
            self,
            model: nn.Module,
            dataloader: DataLoader,
            device: torch.device,
            cfg: Dict[str, Any],
            log_dir: str = "exp",
            incremental_log_dir: bool = False,
            half: bool = False,
            export: bool = False,
            nms_type: str = "nms",
            tta: bool = False,
            tta_scales: List = None,
            tta_flips: List = None,
    ) -> None:
        """Initialize Validator class
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
                            "iou_t": IoU threshold.
                        }
                    }
        :param log_dir: log directory path
        :param incremental_log_dir: use incremental directory.
                                    If set, log_dir will be
                                        {log_dir}/val/{DATE}_runs,
                                        {log_dir}/val/{DATE}_runs1,
                                        {log_dir}/val/{DATE}_runs2,
                                                ...
        :param half: use half precision input
        :param export: export validation results to file
        :param nms_type: NMS type (e.g. nms, batched_nms, fast_nms, matrix_nms)
        """
        super().__init__()
        self.n_class = len(dataloader.dataset.names)  # type: ignore
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.cfg_train = cfg["train"]
        self.cfg_hyp = cfg["hyper_params"]
        self.half = half
        self.export = export
        self.nms_type = nms_type
        self.tta = tta
        self.tta_scales = tta_scales if tta_scales else [1, 0.83, 0.67]
        self.tta_flips = tta_flips if tta_flips else [None, 3, None]

        if incremental_log_dir:
            self.log_dir = increment_path(
                os.path.join(log_dir, "val", datetime.now().strftime("%Y_%m%d_runs"))
            )
        else:
            self.log_dir = log_dir

        if self.export and not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
            logger.info("Export directory: " + colorstr("bold", str(self.log_dir)))

    def convert_target(self, targets: torch.Tensor, width: int, height: int) \
            -> torch.Tensor:
        """Convert targets from normalized coordinates 0.0 ~ 1.0 to pixel coordinates
        :param targets: (n, 6) tensor
                        targets[:, 0] represents index number of the batch
                        targets[:, 1] represents class index number
                        targets[:, 2:] represents normalized xyxy coordinates
        :param width: image width size
        :param height: image height size
        :return converted target tensor
        """
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(
            self.device, non_blocking=True
        )
        return targets

    @abstractmethod
    def validation_step(self, *args: Any, **kwargs: Any) -> Any:
        """Validate a batch."""
        pass

    @abstractmethod
    def validation(self, *args: Any, **kwargs: Any) -> Any:
        """Validate model."""
        pass
