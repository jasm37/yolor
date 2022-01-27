import glob
import math
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from logger import logger

from typing import Dict, List, Optional, Tuple, Union

# Settings START
torch.set_printoptions(linewidth=320, precision=5, profile="long")
np.set_printoptions(
    linewidth=320, formatter={"float_kind": "{:11.5g}".format}
)  # format short g, %precision=5
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ["NUMEXPR_MAX_THREADS"] = str(min(os.cpu_count(), 8))  # NumExpr max threads


# Settings END


def make_divisible(x: int, divisor: int, minimum_check_number: int = 0) -> int:
    """Return 'x' evenly divisible by 'divisor'
    :param x: Input which want to make divisible with 'divisor'
    :param divisor: Divisor
    :param minimum_check_number: Minimum number to check.
    :returns: ceil(x / divisor) * divisor
    """
    if x <= minimum_check_number:
        return math.floor(x)
    else:
        return math.ceil(x / divisor) * divisor


def check_img_size(img_size: int, s: int = 32) -> int:
    """Verify image size is a multiple of stride s
    :param img_size: Current image size
    :param s: Stride
    :returns: New image size verified with stride s
    """
    new_size = make_divisible(img_size, int(s))
    if new_size != img_size:
        logger.warning(
            "WARNING --img-size %g must be multiple of max stride %g, updating to %g"
            % (img_size, s, new_size)
        )
    return new_size


def labels_to_class_weights(labels: Union[list, np.ndarray, torch.Tensor], nc: int = 80) \
        -> torch.Tensor:
    """Get class weights from training labels
    :param labels: labels to use
    :param nc: number of classes
    :return class weight tensor
    """
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()
    _labels = [label for label in labels if len(label) > 0]  # Only use non-empty labels
    c_labels = np.concatenate(_labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = c_labels[:, 0].astype(int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(
        labels: Union[list, np.ndarray],
        nc: int = 80,
        class_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Produce image weights based on class mAPs
    :param labels: labels to use
    :param nc: number of classes
    :param class_weights: class weights
    :return image weight tensor
    """
    if class_weights is None:
        np_class_weights: np.ndarray = np.ones(80)
    else:
        np_class_weights = class_weights

    n = len(labels)
    class_counts = np.array(
        [np.bincount(labels[i][:, 0].astype(int), minlength=nc) for i in range(n)]
    )
    image_weights = (np_class_weights.reshape(1, nc) * class_counts).sum(1)
    return image_weights


def increment_path(path_: str, exist_ok: bool = False, sep: str = "", mkdir: bool = False) \
        -> Path:
    """Increment file or directory path.
    i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    :param path_: path to use increment path
    :param exist_ok: Check if the path already exists and uses the path if exists.
    :param sep: separator string
    :param mkdir: create directory if the path does not exist.
    :return: incremented path.
    """
    path = Path(path_)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix("")
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path

    _dir = path if path.suffix == "" else path.parent  # directory
    if not _dir.exists() and mkdir:
        _dir.mkdir(parents=True, exist_ok=True)  # make directory

    return path


class TimeChecker:
    """Time analyzer class."""

    def __init__(
            self,
            title: str = "",
            ignore_thr: float = 0.05,
            sort: bool = True,
            add_start: bool = True,
            cuda_sync: bool = False,
    ) -> None:
        """Initialize TimeChecker class.
        :param title: name of the time analysis
        :param ignore_thr: time percentage that took below {ignore_thr}% will be ignored for the logging.
        :param sort: log sorted by time consumption ratios
        :param add_start: auto add start time
                         TimeChecker requires at least two time checks.
                         The first time will always be used as the start time.
        :param cuda_sync: Use cuda synchronized time.
        """
        self.times: Dict[str, List[float]] = dict()
        self.name_idx: Dict[str, int] = dict()
        self.idx_name: List[str] = []

        self.title = title
        self.ignore_thr = ignore_thr
        self.sort = sort
        self.cuda_sync = cuda_sync

        if add_start:
            self.add("start")

    def __getitem__(self, name: str) -> Tuple[float, int]:
        """Get time taken.
        :returns:
            time took(s)
            Number of times that {name} event occur.
        """
        idx = self.name_idx[name]
        name_p = self.idx_name[idx - 1]

        times_0 = self.times[name_p]
        times_1 = self.times[name]

        n_time = min(len(times_0), len(times_1))
        time_took = 0.0
        for i in range(n_time):
            time_took += times_1[i] - times_0[i]

        return time_took, n_time

    def add(self, name: str) -> None:
        """Add time point."""
        if self.cuda_sync:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        if name not in self.name_idx:
            self.name_idx[name] = len(self.times)
            self.idx_name.append(name)
            self.times[name] = [time.monotonic()]
        else:
            self.times[name].append(time.monotonic())

    def clear(self) -> None:
        """Clear time records."""
        self.times.clear()
        self.name_idx.clear()
        self.idx_name.clear()

    @staticmethod
    def _convert_unit_str(value: float) -> Tuple[float, str]:
        """Convert second unit to s, ms, ns metric.

        Args:
            value: time(s)
        Returns:
            Converted time value.
            Unit of the time value(s, ms, ns).
        """
        if value < 0.001:
            value *= 1000 * 1000
            unit = "ns"
        elif value < 1:
            value *= 1000
            unit = "ms"
        else:
            unit = "s"

        return value, unit

    @property
    def total_time(self) -> float:
        """Get total time."""
        time_tooks = [self[self.idx_name[i]][0] for i in range(1, len(self.times))]

        return sum(time_tooks)

    def __str__(self) -> str:
        """Convert time checks to the log string."""
        msg = f"[{self.title[-15:]:>15}] "
        time_total = self.total_time
        time_tooks = [self[self.idx_name[i]] for i in range(1, len(self.times))]

        if self.sort:
            idx = np.argsort(np.array(time_tooks)[:, 0])[::-1]
        else:
            idx = np.arange(0, len(self.times) - 1)

        for i in idx:
            time_took = time_tooks[i][0]
            time_ratio = time_took / (time_total + 1e-16)

            time_took, unit = self._convert_unit_str(time_took)

            if time_ratio > self.ignore_thr:
                msg += f"{self.idx_name[i + 1][:10]:>11}: {time_took:4.1f}{unit}({time_ratio * 100:4.1f}%), "

        time_total, unit = self._convert_unit_str(time_total)
        msg += f"{'Total':>11}: {time_total:4.1f}{unit}"
        return msg
