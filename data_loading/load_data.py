import os
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd
import torch

from data_loading.augmentation.augmentation import MultiAugmentationPolicies
from data_loading.data_loaders.cots_data_set import LoadCOTSImagesAndLabels
from data_loading.data_loaders.data_sets import LoadImagesAndLabels
from logger import logger
from utils.general import TimeChecker
from utils.torch_utils import torch_distributed_zero_first

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


def create_dataloader(
        path: str,
        cfg: Dict[str, Any],
        stride: int,
        pad: float = 0.0,
        validation: bool = False,
        preprocess: Optional[Callable] = None,
        prefix: str = "",
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.Dataset]:
    """
    Create YOLO dataset loader
    :param path: root directory of image. The directory structure must follow the rules.
                Ex)  {path} = data/set/images/train
                        dataset/images/train/image001.jpg
                        dataset/images/train/image002.jpg
                        ...

                        dataset/labels/train/image001.txt
                        dataset/labels/train/image002.txt
                        ...

                        dataset/segments/train/image001.txt
                        dataset/segments/train/image002.txt
                        ...
    :param cfg: train_config dictionary
    :param stride: Stride value
    :param pad: padding options for rect
    :param validation: When the data loader is used for validation
    :param preprocess: preprocess function runs in numpy image(CPU).
                       Ex) lambda x: (x / 255.0).astype(np.float32)
    :param prefix: Prefix string for dataset log
    :return: torch DataLoader,
             torch Dataset
    """
    time_checker = TimeChecker(f"{prefix}create")
    rank = LOCAL_RANK if not validation else -1
    batch_size = cfg["train"]["batch_size"] // WORLD_SIZE * (2 if validation else 1)
    workers = cfg["train"]["workers"]

    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(
            path,
            img_size=cfg["train"]["image_size"],
            batch_size=batch_size,
            rect=cfg["train"]["rect"] if not validation else True,  # rectangular training
            label_type=cfg["train"]["label_type"],
            cached_images=cfg["train"]["cache_image"] if not validation else None,
            single_cls=True,
            stride=int(stride),
            pad=pad,
            n_skip=cfg["train"]["n_skip"] if not validation else 0,
            prefix=prefix,
            # image_weights=image_weights,
            yolo_augmentation=cfg["yolo_augmentation"] if not validation else None,
            preprocess=preprocess,
            augmentation=MultiAugmentationPolicies(cfg["augmentation"])
            if not validation
            else None,
        )
    time_checker.add("dataset")

    batch_size = min(batch_size, len(dataset))
    n_workers = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, workers]
    )  # number of workers
    logger.info(f"{prefix}batch_size: {batch_size}, n_workers: {n_workers}")
    sampler: Optional[torch.utils.data.Sampler] = (
        torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    )
    loader = (
        torch.utils.data.DataLoader
        if cfg["train"]["image_weights"]
        else InfiniteDataLoader
    )
    time_checker.add("set_vars")
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        sampler=sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn,
    )
    time_checker.add("dataloader")
    logger.debug(f"{time_checker}")
    return dataloader, dataset


def create_cots_dataloader(
        video_path: str,
        dataframe: pd.DataFrame,
        cfg: Dict[str, Any],
        stride: int,
        pad: float = 0.0,
        validation: bool = False,
        preprocess: Optional[Callable] = None,
        prefix: str = "",
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.Dataset]:  # noqa: data not well referenced in torch.utils
    """
    Create YOLO dataset loader.
    :param video_path: directory path where the videos are stored
    :param dataframe: dataframe with all image paths and its bounding boxes per frame per video
    :param cfg: train_config dictionary
    :param stride: stride value
    :param pad: padding options for rect
    :param validation: When the data loader is used for validation
    :param preprocess: preprocess function runs in numpy image(CPU).
                       Ex) lambda x: (x / 255.0).astype(np.float32)
    :param prefix: Prefix string for dataset log.
    :returns:
        torch DataLoader,
        torch Dataset
    """
    time_checker = TimeChecker(f"{prefix}create")
    rank = LOCAL_RANK if not validation else -1
    batch_size = cfg["train"]["batch_size"] // WORLD_SIZE * (2 if validation else 1)
    workers = cfg["train"]["workers"]

    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadCOTSImagesAndLabels(
            video_path=video_path,
            dataframe=dataframe,
            img_size=cfg["train"]["image_size"],
            batch_size=batch_size,
            label_type=cfg["train"]["label_type"],
            rect=cfg["train"]["rect"],  # if not validation else True,  # rectangular training
            cached_images=cfg["train"]["cache_image"] if not validation else None,
            stride=int(stride),
            pad=pad,
            n_skip=cfg["train"]["n_skip"] if not validation else 0,
            prefix=prefix,
            yolo_augmentation=cfg["yolo_augmentation"] if not validation else None,
            augmentation=MultiAugmentationPolicies(cfg["augmentation"]) if not validation else None,
            preprocess=preprocess,
        )
    time_checker.add("dataset")

    batch_size = min(batch_size, len(dataset))
    n_workers = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, workers]
    )  # number of workers
    logger.info(f"{prefix}batch_size: {batch_size}, n_workers: {n_workers}")
    sampler: Optional[torch.utils.data.Sampler] = (
        torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    )
    loader = (
        torch.utils.data.DataLoader
        if cfg["train"]["image_weights"]
        else InfiniteDataLoader
    )
    time_checker.add("set_vars")
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        sampler=sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn,
    )
    time_checker.add("dataloader")
    logger.debug(f"{time_checker}")
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """Data loader that reuses workers.
    Uses same syntax as torch.utils.data.dataloader.DataLoader.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize InifiniteDataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))  # type: ignore
        self.iterator = super().__iter__()

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.batch_sampler.sampler)  # type: ignore

    def __iter__(self) -> Any:
        """Run iteration."""
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever."""

    def __init__(self, sampler: torch.utils.data.Sampler) -> None:
        """Initialize repeat sampler.

        Args:
            sampler (Sampler)
        """
        self.sampler = sampler

    def __iter__(self) -> Any:
        """Run iteration."""
        while True:
            yield from iter(self.sampler)
