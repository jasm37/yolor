from typing import List, Union, Tuple, Optional

import numpy as np
import torch

DIV_EPS = 1e-16


def resample_segments(segments: List[np.ndarray], n: int = 1000) -> List[np.ndarray]:
    """Interpolate segments by (n, 2) segment points
    :param segments: segmentation coordinates list [(m, 2), ...]
    :param n: number of interpolation
    :return: Interpolated segmentation (n, 2)
    """
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T
        )  # segment xy

    return segments


def segment2box(segment: np.ndarray, width: int = 640, height: int = 640) -> np.ndarray:
    """Convert 1 segment label to 1 box label
    Applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    :param segment: one segmentation coordinates. (n, 2)
    :param width: width constraint
    :param height: height constraint
    :return: bounding box constrained by (width, height)
    """
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return (
        np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))
    )  # xyxy


def segments2boxes(segments: List[np.ndarray]) -> np.ndarray:
    """Convert segment labels to box labels, i.e. (xy1, xy2, ...) to (xywh)
    :param segments: List of segments. [(n1, 2), (n2, 2), ...]
    :return: box labels (n, 4)
    """
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes), clip_eps=None, check_validity=False)  # type: ignore


def box_candidates_mask(
        box1: np.ndarray,
        box2: np.ndarray,
        wh_thr: float = 2,
        ar_thr: float = 20,
        area_thr: float = 0.1,
) -> np.ndarray:  # box1(4,n), box2(4,n)
    """Compute candidate boxes
    :param box1: before augment
    :param box2: after augment
    :param wh_thr: width and height threshold (pixels),
    :param ar_thr: aspect ratio threshold
    :param area_thr: area_ratio
    :return: Boolean mask index of the box candidates.
    """
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + DIV_EPS), h2 / (w2 + DIV_EPS))  # aspect ratio
    return (
            (w2 > wh_thr)
            & (h2 > wh_thr)
            & (w2 * h2 / (w1 * h1 + DIV_EPS) > area_thr)
            & (ar < ar_thr)
    )  # candidates


def clip_coords(
        boxes: Union[torch.Tensor, np.ndarray],
        wh: Tuple[float, float],
        inplace: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """Clip bounding boxes with xyxy format to given wh (width, height).
    :param boxes: bounding boxes (n, 4) (x1, y1, x2, y2)
    :param wh: image size (width, height)
    :param inplace: inplace modification
    :return clipped boxes
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        if not inplace:
            boxes = boxes.clone()

        boxes[:, 0].clamp_(0, wh[0])  # x1
        boxes[:, 1].clamp_(0, wh[1])  # y1
        boxes[:, 2].clamp_(0, wh[0])  # x2
        boxes[:, 3].clamp_(0, wh[1])  # y2
    else:  # np.array (faster grouped)
        if not inplace:
            boxes = np.copy(boxes)

        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, wh[0])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, wh[1])  # y1, y2

    return boxes


def xyn2xy(
        x: Union[torch.Tensor, np.ndarray],
        wh: Tuple[float, float] = (640, 640),
        pad: Tuple[float, float] = (0.0, 0.0),
) -> Union[torch.Tensor, np.ndarray]:
    """Convert normalized xy (n, 2) to pixel coordinates xy
    :param x: input segments
    :param wh: Image size (width and height). If normalized xywh to pixel xyxy format, place image size here
    :param pad: image padded size (width and height)
    """
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = wh[0] * x[:, 0] + pad[0]  # top left x
    y[:, 1] = wh[1] * x[:, 1] + pad[1]  # top left y
    return y


def xyxy2xywh(
        x: Union[torch.Tensor, np.ndarray],
        wh: Tuple[float, float] = (1.0, 1.0),
        pad: Tuple[float, float] = (0.0, 0.0),
        clip_eps: Optional[float] = None,
        check_validity: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """Convert (n, 4) bound boxes from xyxy to xywh format
    [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    :param x: (n, 4) xyxy coordinates
    :param wh: image size (width, height)
               Give image size only if you want to
               normalized pixel coordinates to normalized coordinates
    :param pad: image padded size (width and height)
    :param clip_eps: clip coordinates by wh with epsilon margin. If clip_eps is not None
                     epsilon value is recommended to be 1E-3
    :param check_validity: bounding box width and height validity check
                          which make bounding boxes to the following conditions
                          1) (x1 - width / 2) >= 0
                          2) (y1 - height / 2) >= 0
                          3) (x2 + width / 2) <= 1
                          4) (y2 + height / 2) <= 1
    :return: (xywh / wh) with centered xy
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)

    if clip_eps is not None:
        y = clip_coords(y, (wh[0] - clip_eps, wh[1] - clip_eps))

    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / wh[0] + pad[0]  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / wh[1] + pad[1]  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / wh[0]  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / wh[1]  # height

    if check_validity:
        y[:, 2] = y[:, 2] + (np.minimum((y[:, 0] - (y[:, 2] / 2)), 0) * 2)
        y[:, 2] = y[:, 2] - ((np.maximum((y[:, 0] + (y[:, 2] / 2)), 1) - 1) * 2)
        y[:, 3] = y[:, 3] + (np.minimum((y[:, 1] - (y[:, 3] / 2)), 0) * 2)
        y[:, 3] = y[:, 3] - ((np.maximum((y[:, 1] + (y[:, 3] / 2)), 1) - 1) * 2)

        y = y.clip(1e-12, 1)

    return y


def xywh2xyxy(
        x: Union[torch.Tensor, np.ndarray],
        ratio: Tuple[float, float] = (1.0, 1.0),
        wh: Tuple[float, float] = (1.0, 1.0),
        pad: Tuple[float, float] = (0.0, 0.0),
) -> Union[torch.Tensor, np.ndarray]:
    """Convert (n, 4) bound boxes from xywh to xyxy format
    [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    :param x: (n, 4) xywh coordinates
    :param ratio: label ratio adjustment. Default value won't change anything other than xywh to xyxy
    :param wh: Image size (width and height). If normalized xywh to pixel xyxy format, place image size here
    :param pad: image padded size (width and height)
    :return:  (ratio * wh * xyxy + pad)
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ratio[0] * wh[0] * (x[:, 0] - x[:, 2] / 2) + pad[0]  # top left x
    y[:, 1] = ratio[1] * wh[1] * (x[:, 1] - x[:, 3] / 2) + pad[1]  # top left y
    y[:, 2] = ratio[0] * wh[0] * (x[:, 0] + x[:, 2] / 2) + pad[0]  # bottom right x
    y[:, 3] = ratio[1] * wh[1] * (x[:, 1] + x[:, 3] / 2) + pad[1]  # bottom right y
    return y


def scale_coords(
        img1_shape: Tuple[float, float],
        coords: Union[torch.Tensor, np.ndarray],
        img0_shape: Tuple[float, float],
        ratio_pad: Optional[Union[tuple, list, np.ndarray]] = None,
) -> Union[torch.Tensor, np.ndarray]:
    """Rescale coords (xyxy) from img1_shape to img0_shape
    :param img1_shape: current image shape (h, w)
    :param coords: (xyxy) coordinates
    :param img0_shape: target image shape (h, w)
    :param ratio_pad: padding ratio  (w, h)
    :return scaled coordinates
    """
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape[::-1])  # clip_coord use wh image shape.
    return coords
