import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from data_loading.augmentation.label_augmentation import ImgAugmentator as Aug
from data_loading.label_adapters import xywh2xyxy, xyxy2xywh, xyn2xy
from data_loading.parsers.csv_parser import CsvParser
from data_loading.data_loaders.data_sets import LoadImages, letterbox


class LoadCOTSImagesAndLabels(LoadImages):
    def __init__(
            self,
            video_path: str,
            dataframe: pd.DataFrame,
            img_size: int = 1280,
            batch_size: int = 16,
            label_type: str = "segments",
            image_weights: bool = False,
            stride: int = 32,
            rect: bool = False,
            cached_images: Optional[str] = None,
            pad: float = 0.0,
            n_skip: int = 0,
            prefix: str = "",
            yolo_augmentation: Optional[Dict[str, Any]] = None,
            preprocess: Optional[Callable] = None,
            augmentation: Optional[Callable] = None,
    ):
        """
        :param video_path: video directory
        :param dataframe: image path, label dataframe
        :param img_size: Minimum width or height size
        :param batch_size: Batch size
        :param rect: use rectangular image
        :param label_type: label directory name. This should be either 'labels' or 'segments'
        :param cached_images: use caching images. if None, caching will not be used
                            'mem': Caching images in memory
                            'disk': Caching images in disk
        :param image_weights: image weights
        :param stride: Stride value
        :param pad: pad size for rectangular image. This applies only when rect is True
        :param n_skip: Skip n images per one image
                       Ex) If we have 1024 images and n_skip is 1, then total 512 images will be used
        :param yolo_augmentation: augmentation parameters for YOLO augmentation
        :param prefix: logging prefix message
        :param preprocess: preprocess function which takes (x: np.ndarray) and returns (np.ndarray)
        :param yolo_augmentation: yolo augmentations
        :param augmentation: augmentation function which takes (x: np.ndarray, label: np.ndarray)
                            and returns (np.ndarray, np.ndarray), the label format is xyxy with pixel coordinates

        """
        with CsvParser(video_path, dataframe=dataframe) as parser:
            img_files, labels, shapes = parser.read_all(return_shape=True)

        super().__init__(
            path="",
            img_size=img_size,
            batch_size=batch_size,
            rect=rect,
            cached_images=cached_images,
            stride=stride,
            pad=pad,
            n_skip=n_skip,
            prefix=prefix,
            preprocess=preprocess,
            augmentation=augmentation,
            image_files=img_files,
            img_shapes=shapes
        )

        self.labels = labels
        self.segments = [[] for _ in self.labels]
        self.label_type = label_type
        self.names = ["starfish"]
        self.yolo_augmentation = (
            yolo_augmentation if yolo_augmentation is not None else {}
        )
        self.image_weights = image_weights

    def __getitem__(
            self, index: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        str,
        Tuple[Tuple[int, int], Tuple[Tuple[float, float], Tuple[float, float]]],
    ]:
        """Get item from given index
        :param index: Index number for the image
        :return:
            PyTorch image (CHW),
            Normalized(0.0 ~ 1.0) xywh labels,
            Image path,
            Image shapes (Original, (ratio(new/original), pad(h,w)))
        """
        index = self.indices[index]

        shape = (
            self.batch_shapes[self.batch_idx[index]]
            if self.rect
            else (self.img_size, self.img_size)
        )

        if random.random() < self.yolo_augmentation.get("mosaic", 0.0):
            img, labels = self._load_mosaic9(index)
            shapes = (0, 0), ((0.0, 0.0), (0.0, 0.0))
            if random.random() < self.yolo_augmentation.get("mixup", 1.0):
                img, labels = Aug.mixup(
                    img,
                    labels,
                    *self._load_mosaic9(random.randint(0, len(self.img_files) - 1)),
                )

        else:
            img, (h0, w0), (h1, w1) = self._load_image(index)

            img, ratio, pad = letterbox(
                img,
                stride=self.stride,
                new_shape=shape,
                auto=False,
                scale_fill=False,
                scale_up=self.yolo_augmentation.get("augment", False),
            )
            shapes = (h0, w0), ((h1 / h0, w1 / w0), pad)

            if self.labels[index].shape[0] == 0:
                labels = np.empty((0, 5), dtype=np.float32)
                segments = []
            else:
                labels = self.labels[index].copy()
                segments = self.segments[index].copy()

            # Adjust bboxes to the letterbox.
            if labels.size:
                labels[:, 1:] = xywh2xyxy(
                    labels[:, 1:], ratio=ratio, wh=(w1, h1), pad=pad
                )
                segments = [xyn2xy(x, wh=(w1, h1), pad=pad) for x in segments]  # noqa

            # Copy-paste 2
            copy_paste_cfg = self.yolo_augmentation.get("copy_paste2", {})
            if copy_paste_cfg.get("p", 0.0) > 0.0:
                for _ in range(copy_paste_cfg.get("n_img", 3)):
                    img, labels, segments = self._load_copy_paste(
                        img=img, labels=labels, seg=segments
                    )

            if self.yolo_augmentation.get("augment", False):
                img, labels = Aug.random_perspective(
                    img,
                    labels,
                    degrees=self.yolo_augmentation.get("degrees", 0.0),
                    translate=self.yolo_augmentation.get("translate", 0.1),
                    scale=self.yolo_augmentation.get("scale", 0.5),
                    shear=self.yolo_augmentation.get("shear", 0.0),
                    perspective=self.yolo_augmentation.get("perspective", 0.0),
                )  # border to remove

        # Normalize bboxes
        if labels.size:
            labels[:, 1:] = xyxy2xywh(
                labels[:, 1:], wh=img.shape[:2][::-1], clip_eps=1e-3
            )

        if self.augmentation:
            img, labels = self.augmentation(img, labels)

            if self.yolo_augmentation.get("augment", False):
                Aug.augment_hsv(
                    img,
                    self.yolo_augmentation.get("hsv_h", 0.015),
                    self.yolo_augmentation.get("hsv_s", 0.7),
                    self.yolo_augmentation.get("hsv_v", 0.4),
                )

        if self.preprocess:
            img = self.preprocess(img)

        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        n_labels = len(labels)
        labels_out = torch.zeros((n_labels, 6))
        if n_labels > 0:
            labels_out[:, 1:] = torch.from_numpy(labels)
        torch_img = torch.from_numpy(img)

        return torch_img, labels_out, self.img_files[index], shapes

    def _load_mosaic4(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates a 4 image-mosaic given the base image index
        :param index: base image index
        :return:
            mosaic image from the base image and 3 others
            mosaic labels from all joined images
        """
        img_half_size = self.img_size // 2
        # height, width
        mosaic_center = [
            int(random.uniform(img_half_size, 2 * self.img_size - img_half_size))
            for _ in range(2)
        ]

        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)

        loaded_imgs = [self._load_image(idx) for idx in indices]

        mosaic_img = np.full(
            (
                int(self.img_size * 2),
                int(self.img_size * 2),
                loaded_imgs[0][0].shape[2],
            ),
            114,
            dtype=np.uint8,
        )
        mosaic_labels = []
        mosaic_segments = []

        mc_h, mc_w = mosaic_center
        assert len(loaded_imgs) == 4  # This must be the case for the following loop to work properly
        for i, (img, _, (h, w)) in enumerate(loaded_imgs):
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(mc_w - w, 0), max(mc_h - h, 0), mc_w, mc_h
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = (
                    mc_w,
                    max(mc_h - h, 0),
                    min(mc_w + w, self.img_size * 2),
                    mc_h,
                )
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = (
                    max(mc_w - w, 0),
                    mc_h,
                    mc_w,
                    min(self.img_size * 2, mc_h + h),
                )
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = (
                    mc_w,
                    mc_h,
                    min(mc_w + w, self.img_size * 2),
                    min(self.img_size * 2, mc_h + h),
                )
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # noqa: these values are assigned in each if-case
            pad_w = x1a - x1b
            pad_h = y1a - y1b

            if self.labels[indices[i]].shape[0] == 0:
                labels = np.empty((0, 5), dtype=np.float32)
                segments = []
            else:
                labels = self.labels[indices[i]].copy()
                segments = self.segments[indices[i]].copy()

            if labels.size:
                labels[:, 1:] = xywh2xyxy(labels[:, 1:], wh=(w, h), pad=(pad_w, pad_h))
                segments = [xyn2xy(x, wh=(w, h), pad=(pad_w, pad_h)) for x in segments]
            mosaic_labels.append(labels)
            mosaic_segments.extend(segments)

        mosaic_labels_np = np.concatenate(mosaic_labels, 0)

        for x in (mosaic_labels_np[:, 1:], *mosaic_segments):
            np.clip(x, 1e-3, 2 * self.img_size, out=x)

        mosaic_img, mosaic_labels_np, mosaic_segments = Aug.copy_paste(
            mosaic_img,
            mosaic_labels_np,
            mosaic_segments,
            prob=self.yolo_augmentation.get("copy_paste", 0.0),
        )

        # Copy-paste 2
        copy_paste_cfg = self.yolo_augmentation.get("copy_paste2", {})
        if copy_paste_cfg.get("p", 0.0) > 0.0:
            for _ in range(copy_paste_cfg.get("n_img", 3)):
                mosaic_img, mosaic_labels_np, mosaic_segments = self._load_copy_paste(
                    mosaic_img, mosaic_labels_np, mosaic_segments
                )

        mosaic_img, mosaic_labels_np = Aug.random_perspective(
            mosaic_img,
            mosaic_labels_np,
            mosaic_segments,
            degrees=self.yolo_augmentation.get("degrees", 0.0),
            translate=self.yolo_augmentation.get("translate", 0.1),
            scale=self.yolo_augmentation.get("scale", 0.5),
            shear=self.yolo_augmentation.get("shear", 0.0),
            perspective=self.yolo_augmentation.get("perspective", 0.0),
            border=(-img_half_size, -img_half_size),
        )  # border to remove

        return mosaic_img, mosaic_labels_np

    def _load_mosaic9(self, index):
        """
        Creates a 9 image-mosaic given the base image index
        :param index: base image index
        :return:
          mosaic image from the base image and 8 others
          mosaic labels from all joined images
        """
        img_half_size = self.img_size // 2
        indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
        random.shuffle(indices)
        loaded_imgs = [self._load_image(idx) for idx in indices]
        s = self.img_size
        mosaic_img = np.full(
            (
                int(s * 3),
                int(s * 3),
                loaded_imgs[0][0].shape[2],
            ),
            114,
            dtype=np.uint8,
        )
        mosaic_labels_np = []
        mosaic_segments = []

        assert len(loaded_imgs) == 9  # This must be the case for the following loop to work properly
        for i, (img, _, (h, w)) in enumerate(loaded_imgs):
            if i == 0:  # center
                mosaic_img = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Labels
            if self.labels[indices[i]].shape[0] == 0:
                labels = np.empty((0, 5), dtype=np.float32)
                segments = []
            else:
                labels = self.labels[indices[i]].copy()
                segments = self.segments[indices[i]].copy()

            if labels.size:
                labels[:, 1:] = xywh2xyxy(labels[:, 1:], wh=(w, h),
                                          pad=(padx, pady))  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, wh=(w, h), pad=(padx, pady)) for x in segments]
            mosaic_labels_np.append(labels)
            mosaic_segments.extend(segments)

            # Image
            mosaic_img[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = (int(random.uniform(0, s)) for _ in (-img_half_size, -img_half_size))  # mosaic center x, y
        mosaic_img = mosaic_img[yc:yc + 2 * s, xc:xc + 2 * s]

        # Concat/clip labels
        mosaic_labels_np = np.concatenate(mosaic_labels_np, 0)
        mosaic_labels_np[:, [1, 3]] -= xc
        mosaic_labels_np[:, [2, 4]] -= yc
        c = np.array([xc, yc])  # centers
        mosaic_segments = [x - c for x in mosaic_segments]

        for x in (mosaic_labels_np[:, 1:], *mosaic_segments):
            np.clip(x, 0, 2 * s, out=x)

        mosaic_img, mosaic_labels_np, mosaic_segments = Aug.copy_paste(
            mosaic_img,
            mosaic_labels_np,
            mosaic_segments,
            prob=self.yolo_augmentation.get("copy_paste", 0.0),
        )

        # Copy-paste 2
        copy_paste_cfg = self.yolo_augmentation.get("copy_paste2", {})
        if copy_paste_cfg.get("p", 0.0) > 0.0:
            for _ in range(copy_paste_cfg.get("n_img", 3)):
                mosaic_img, mosaic_labels_np, mosaic_segments = self._load_copy_paste(
                    mosaic_img, mosaic_labels_np, mosaic_segments
                )

        mosaic_img, mosaic_labels_np = Aug.random_perspective(
            mosaic_img,
            mosaic_labels_np,
            mosaic_segments,
            degrees=self.yolo_augmentation.get("degrees", 0.0),
            translate=self.yolo_augmentation.get("translate", 0.1),
            scale=self.yolo_augmentation.get("scale", 0.5),
            shear=self.yolo_augmentation.get("shear", 0.0),
            perspective=self.yolo_augmentation.get("perspective", 0.0),
            border=(-img_half_size, -img_half_size),  # border to remove
        )
        return mosaic_img, mosaic_labels_np

    def _load_copy_paste(
            self, img: np.ndarray, labels: np.ndarray, seg: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Load copy paste augmentation
        This method will copy paste objects from another image file
        :param img: input image
        :param labels: image labels
        :param seg: image object segmentations
        :returns:
            Copy-pasted image
            Copy-pasted labels
            Copy-pasted segmentations
        """
        img_idx_for_copy = random.choice(self.indices)
        img_for_copy, _, (h, w) = self._load_image(img_idx_for_copy)

        if self.labels[img_idx_for_copy].shape[0] == 0:
            labels_for_copy = np.empty((0, 5), dtype=np.float32)
            seg_for_copy = []
        else:
            labels_for_copy = self.labels[img_idx_for_copy].copy()
            seg_for_copy = self.segments[img_idx_for_copy].copy()

        if labels_for_copy.size:
            labels_for_copy[:, 1:] = xywh2xyxy(
                labels_for_copy[:, 1:], wh=(w, h), pad=(0, 0)
            )
            seg_for_copy = [xyn2xy(x, wh=(w, h), pad=(0, 0)) for x in seg_for_copy]  # noqa: it would remain empty

        copy_paste_cfg = (
            self.yolo_augmentation["copy_paste2"]
            if "copy_paste2" in self.yolo_augmentation.keys()
            else {}
        )

        copy_paste_img, copy_paste_label, copy_paste_seg = Aug.copy_paste2(
            im1=img,
            labels1=labels,
            seg1=seg,
            im2=img_for_copy,
            labels2=labels_for_copy,
            seg2=seg_for_copy,
            scale_min=copy_paste_cfg.get("scale_min", 0.9),
            scale_max=copy_paste_cfg.get("scale_max", 1.1),
            prob=copy_paste_cfg.get("p", 0.0),
            area_thr=copy_paste_cfg.get("area_thr", 10),
            ioa_thr=copy_paste_cfg.get("ioa_thr", 0.3),
        )

        return copy_paste_img, copy_paste_label, copy_paste_seg

    @staticmethod
    def collate_fn(
            batch: List[
                Tuple[
                    torch.Tensor, torch.Tensor, str, Tuple[Tuple[int, int], Tuple[int, int]]
                ]
            ]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Tuple[str, ...],
        Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...],
    ]:
        """
        Handle Collate in PyTorch.
        :param batch: collated batch item.
        """
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img, 0), torch.cat(label, 0), path, shapes
