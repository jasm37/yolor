import random
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as alb
import numpy as np
from albumentations import DualTransform

from logger import logger


class BoxJitter(DualTransform):  # noqa
    """Apply bbox jitter"""

    def __init__(self, always_apply: bool = False, p: float = 1, jitter: float = 0.01, ) -> None:
        """
        Initialize BoxJitter augmentation
        :param always_apply: whether to always apply this augmentation.
        :param p: probability to run this augmentation
        :param jitter: Maximum jitter size of the bounding box
                       i.e. 0.01 = 1% of width or height jitter
        """
        super().__init__(always_apply, p)
        self.jitter = jitter

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Pass original image."""
        return img

    def get_params(self) -> Dict[str, float]:
        """Get bbox jitter parameters."""
        return {
            "x1_jitter": random.uniform(-self.jitter, self.jitter),
            "y1_jitter": random.uniform(-self.jitter, self.jitter),
            "x2_jitter": random.uniform(-self.jitter, self.jitter),
            "y2_jitter": random.uniform(-self.jitter, self.jitter),
        }

    def apply_to_bbox(self, bbox: Tuple[float, float, float, float], **params: Any) \
            -> Tuple[float, float, float, float]:
        """
        Add jitter to bounding boxes
        :param bbox: bbox to be augmented as [x1, y1, x2, y2]
        :param params: jitter proportion parameters per coordinate
        :return: bbox with jitter applied to it
        """
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return (
            max(min(bbox[0] + (params["x1_jitter"] * width), 1.0), 0.0),
            max(min(bbox[1] + (params["y2_jitter"] * height), 1.0), 0.0),
            max(min(bbox[2] + (params["x2_jitter"] * width), 1.0), 0.0),
            max(min(bbox[3] + (params["y2_jitter"] * height), 1.0), 0.0),
        )


class AugmentationPolicy:
    """Augmentation policy with albumentation."""

    def __init__(self, policy: Dict[str, Dict], p: float = 1.0) -> None:
        """
        Augmentation with albumentation
        :param policy: augmentation policy described in dictionary format
                      each key name represents albumentations.{KEY} augmentation and
                      value contains keyword arguments for the albumentations.{KEY}
                      i.e. {"Blur": {"p": 0.5},
                           "Flip": {"p": 0.5}
                           }
        :param p: probability to run this augmentation policy
        """
        self.prob = p
        aug_module_paths = ["albumentations", __name__]

        transforms = []
        for aug_name, kwargs in policy.items():
            found_module = False
            for module_path in aug_module_paths:
                if hasattr(__import__(module_path, fromlist=[""]), aug_name):
                    transforms.append(
                        getattr(__import__(module_path, fromlist=[""]), aug_name)(
                            **kwargs
                        )
                    )
                    found_module = True
                    break
            if not found_module:
                logger.warning(
                    f"Can not find {aug_name} augmentation. {aug_name} will not be used."
                )

        self.transform = alb.Compose(
            transforms,
            bbox_params=alb.BboxParams(format="yolo", label_fields=["class_labels"]),
        )

    def __call__(self, img: np.ndarray, labels: Optional[np.ndarray] = None) \
            -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Augmentation function with label(optional)
        :param img: image (HWC) to augment with (0 ~ 255) range
        :param labels: (n, 5) labels. (class_id, x1, y1, x2, y2) with pixel coordinates
        :return: augmented image if labels is None else (augmented image, labels)
        """
        aug_labels = np.array(((0, 0.1, 0.1, 0.1, 0.1),)) if labels is None else labels

        if random.random() < self.prob:
            augmented = self.transform(
                image=img, bboxes=aug_labels[:, 1:], class_labels=aug_labels[:, 0]
            )
            im = augmented["image"]
            if len(augmented["class_labels"]) > 0:
                aug_labels = np.hstack(
                    [
                        np.array(augmented["class_labels"]).reshape(-1, 1),
                        np.array(augmented["bboxes"]),
                    ]
                )
        else:
            im = img

        if labels is not None:
            return im, aug_labels
        else:
            return im


class MultiAugmentationPolicies:
    """Multiple augmentation policies with albumentations."""

    def __init__(self, policies: List[Dict]) -> None:
        """
        Multiple augmentation with albumentation
        :param policies: List of augmentation policies described in dictionary format
                        each key name represents albumentations.{KEY} augmentation and
                        value contains keyword arguments for the albumentations.{KEY}
                        i.e. [
                                {
                                "policy":
                                    {
                                        "Blur": {"p": 0.5},
                                         "Flip": {"p": 0.5}
                                    },
                                "p": 0.3
                                },
                                {
                                "policy":
                                    {
                                        "RandomGamma": {"p": 0.5},
                                         "HorizontalFlip": {"p": 0.5}
                                    },
                                "p": 0.3
                                }
                            ]
        """
        self.transforms = [
            AugmentationPolicy(aug["policy"], aug["p"]) for aug in policies
        ]

    def __call__(self, img: np.ndarray, labels: Optional[np.ndarray] = None) \
            -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply multiple augmentation policy with label(optional)
        :param img: image (HWC) to augment with (0 ~ 255) range
        :param labels: (n, 5) labels. (class_id, x1, y1, x2, y2) with pixel coordinates
        :return: augmented image if labels is None else (augmented image, labels)
        """
        for transform in self.transforms:
            if labels is not None:
                img, labels = transform(img, labels)
            else:
                img = transform(img)  # type: ignore

        return img, labels if labels is not None else img
