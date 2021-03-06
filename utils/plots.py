import math
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Sequence

from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

from data_loading.label_adapters import xywh2xyxy
from utils.constants import PLOT_COLOR


def hist2d(x: np.ndarray, y: np.ndarray, n: int = 100) -> np.ndarray:
    """Draw 2D histogram.
    :param x: x values.
    :param y: y values.
    :param n: linspace step.
    :returns: a numpy array which contains histogram 2d.
    """
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)  # noqa
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def plot_one_box(
        x: np.ndarray,
        img: np.ndarray,
        color: Union[Tuple[int, int, int], List[int]] = None,
        label: str = None,
        line_thickness: float = None,
        alpha: float = 1.,
) -> None:
    """Plot one bounding box on image
    :param x: box coordinates
    :param img: base image to plot label
    :param color: box edge color
    :param label: label to plot
    :param line_thickness: line thickness
    :param alpha: box transparency alpha
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    _buffer = img.copy()
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(_buffer, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(_buffer, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            _buffer,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
    cv2.addWeighted(_buffer, alpha, img, 1 - alpha, 0, img)


def plot_label_histogram(labels: np.ndarray, save_dir: Union[str, Path] = "") -> None:
    """Plot dataset labels
    :param labels: image labels
    :param save_dir: save directory for saving the histogram
    """
    c, b = labels[:, 0], labels[:, 1:].transpose()
    nc = int(c.max() + 1)  # number of classes

    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0].set_xlabel("classes")
    ax[1].scatter(b[0], b[1], c=hist2d(b[0], b[1], 90), cmap="jet")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[2].scatter(b[2], b[3], c=hist2d(b[2], b[3], 90), cmap="jet")
    ax[2].set_xlabel("width")
    ax[2].set_ylabel("height")
    plt.savefig(Path(save_dir) / "labels.png", dpi=200)
    plt.close()

    # seaborn correlogram
    try:
        import pandas as pd
        import seaborn as sns

        x = pd.DataFrame(b.transpose(), columns=["x", "y", "width", "height"])
        sns.pairplot(
            x,
            corner=True,
            diag_kind="hist",
            kind="scatter",
            markers="o",
            plot_kws=dict(s=3, edgecolor=None, linewidth=1, alpha=0.02),
            diag_kws=dict(bins=50),
        )
        plt.savefig(Path(save_dir) / "labels_correlogram.png", dpi=200)
        plt.close()
    except Exception:
        pass


def format_and_plot_boxes(
        arrays: np.ndarray,
        class_id: int,
        block_x: float,
        block_y: float,
        denormalize: bool,
        w: int,
        h: int,
        names: list,
        line_width: int,
        color_list: List[tuple],
        mosaic: np.ndarray,
        conf_thresh: float = 0.3,
        prefix: str = "",
        alpha: float = 1.
) -> None:
    """
    Format detections and plot them into the given mosaic image
    :param arrays: detection array
    :param class_id: object/class id
    :param block_x: block-x to select position in the mosaic
    :param block_y: block-y to select position in the mosaic
    :param denormalize: whether to denormalize the boxes
    :param w: image width
    :param h: image_height
    :param names: class names
    :param line_width: box line width
    :param color_list: color list for the classes
    :param mosaic: mosaic image where to plot into
    :param conf_thresh: minimum threshold to plot detections
    :param prefix: prefix to prepend to class names in the image
    :param alpha: transparency alpha for the boxes and text
    """
    image_preds = arrays[arrays[:, 0] == class_id]
    boxes = xywh2xyxy(image_preds[:, 2:6]).T
    classes = image_preds[:, 1].astype("int")  # type: ignore
    gt = image_preds.shape[1] == 6  # ground truth if no conf column
    conf: Optional[np.ndarray] = (
        None if gt else image_preds[:, 6]
    )  # check for confidence presence (gt vs pred)
    if denormalize:
        boxes[[0, 2]] *= w
        boxes[[1, 3]] *= h
    boxes[[0, 2]] += block_x
    boxes[[1, 3]] += block_y
    for j, box in enumerate(boxes.T):
        cls = int(classes[j])
        _color = color_list[cls % len(color_list)]
        cls = names[cls] if names else cls

        if gt or conf[j] > conf_thresh:
            label = prefix + "%s" % cls if gt else "%s %.1f" % (cls, conf[j])  # type: ignore
            plot_one_box(
                box, mosaic, label=label, color=_color, line_thickness=line_width, alpha=alpha
            )


def plot_images(
        images: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray] = np.array([]),
        predictions: Union[Sequence[torch.Tensor], Sequence[np.ndarray]] = np.array([]),
        unnormalize: bool = True,
        paths: Optional[Sequence[str]] = None,
        fname: str = "images.jpg",
        names: Optional[Union[tuple, list]] = None,
        max_size: int = 4800,
        max_subplots: int = 16,
        resize_mosaic: bool = True
) -> np.ndarray:
    """Plot images."""
    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()

    if isinstance(targets, torch.Tensor):
        np_targets = targets.cpu().numpy()
    else:
        np_targets = targets

    if len(predictions) > 0 and isinstance(predictions[0], torch.Tensor):
        np_predictions = [_pred.cpu().numpy() for _pred in predictions]  # type: Sequence[np.ndarray]
    else:
        np_predictions = predictions

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # Empty array for output
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

    # Fix class - colour map
    prop_cycle = plt.rcParams["axes.prop_cycle"]

    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb

    def hex2rgb(h):  # noqa
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))  # noqa

    # hex2rgb = lambda h: tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()["color"]]

    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y: block_y + h, block_x: block_x + w, :] = img  # noqa
        if len(np_targets) > 0:
            format_and_plot_boxes(np_targets, i, block_x, block_y,
                                  unnormalize, w, h, names, tl, color_lut, mosaic, prefix="t")

        if len(np_predictions) > 0 and len(np_predictions[i]) > 0:
            format_and_plot_boxes(np_predictions[i], i, block_x, block_y,
                                  unnormalize, w, h, names, tl, color_lut[::-1], mosaic, prefix="p", alpha=0.6)

        # Draw image filename labels
        if paths is not None:
            label = os.path.basename(paths[i])[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(
                mosaic,
                label,
                (block_x + 5, block_y + t_size[1] + 5),
                0,
                tl / 3,
                [220, 220, 220],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )

        # Image border
        cv2.rectangle(
            mosaic,
            (block_x, block_y),
            (block_x + w, block_y + h),
            (255, 255, 255),
            thickness=3,
        )

    if fname is not None:
        if resize_mosaic:
            mosaic = cv2.resize(
                mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation=cv2.INTER_AREA
            )
        cv2.imwrite(str(fname), cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))

    return mosaic


def draw_labels(
        img: np.ndarray,
        label_list: np.ndarray,
        label_info: Dict[int, str],
        norm_xywh: bool = True,
) -> np.ndarray:
    """Draw label information on the image.
    :param img: image to draw labels
    :param label_list: (n, 5) label information of img with normalized xywh format.
                       (class_id, centered x, centered y, width, height)
    :param label_info: label names. Ex) {0: 'Person', 1:'Car', ...}
    :param norm_xywh: Flag for label as normalized (0.0 ~ 1.0) xywh format.
                    Otherwise, label will be considered as pixel wise xyxy format.
    :returns: label drawn image.
    """
    overlay_alpha = 0.3
    label_list = np.copy(label_list)
    if norm_xywh:
        label_list[:, 1:] = xywh2xyxy(
            label_list[:, 1:], wh=(float(img.shape[1]), float(img.shape[0]))
        )

    for label in label_list:
        class_id = int(label[0])
        class_str = label_info[class_id]

        xy1 = tuple(label[1:3].astype("int"))
        xy2 = tuple(label[3:5].astype("int"))
        plot_color = tuple(map(int, PLOT_COLOR[class_id]))

        overlay = img.copy()
        overlay = cv2.rectangle(overlay, xy1, xy2, plot_color, -1)
        img = cv2.addWeighted(overlay, overlay_alpha, img, 1 - overlay_alpha, 0)
        img = cv2.rectangle(img, xy1, xy2, plot_color, 1)

        (text_width, text_height), baseline = cv2.getTextSize(class_str, 3, 0.5, 1)
        overlay = img.copy()
        overlay = cv2.rectangle(
            overlay,
            (xy1[0], xy1[1] - text_height),
            (xy1[0] + text_width, xy1[1]),
            (plot_color[0] // 0.3, plot_color[1] // 0.3, plot_color[2] // 0.3),
            -1,
        )
        img = cv2.addWeighted(overlay, overlay_alpha + 0.2, img, 0.8 - overlay_alpha, 0)
        cv2.putText(
            img,
            class_str,
            xy1,
            3,
            0.5,
            (plot_color[0] // 3, plot_color[1] // 3, plot_color[2] // 3),
            1,
            cv2.LINE_AA,
        )
    return img


def plot_pr_curve(
        px: np.ndarray,
        py: list,
        ap: np.ndarray,
        save_dir: Union[str, Path] = "pr_curve.png",
        names: list = None,  # noqa: B006

) -> None:
    """Plot precision-recall curve.
    :param px: x axis for plotting.
    :param py: y axis for plotting.
    :param ap: average precision.
    :param save_dir: save directory for plotted image.
    :param names: class names.
    """
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    n_py = np.stack(py, axis=1)
    names = names or []
    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(n_py.T):
            ax.plot(
                px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}"
            )  # plot(recall, precision)
    else:
        ax.plot(px, n_py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(
        px,
        n_py.mean(1),
        linewidth=3,
        color="blue",
        label="all classes %.3f mAP@0.5" % ap[:, 0].mean(),
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()


def plot_mc_curves(
        px: np.ndarray,
        py: Dict[float, np.ndarray],
        iou_values: np.ndarray,
        save_dir: Union[str, Path] = "mc_curve.png",
        names: list = None,  # noqa: B006
        xlabel: str = "Confidence",
        ylabel: str = "Metric",
):
    """Plot metric-confidence curve
    :param px: x axis for plotting
    :param py: y axis for plotting
    :param iou_values: IoU values for which each subplot belong to
    :param save_dir: save directory for plotted image
    :param names: class names
    :param xlabel: x axis label
    :param ylabel: y axis label
    """
    num_cases = len(py)
    grid_side = math.ceil(math.sqrt(num_cases))
    length, rest = divmod(num_cases, grid_side)
    fig = plt.figure(constrained_layout=True, figsize=(4 * grid_side, 4 * grid_side))
    spec = GridSpec(length + 1, grid_side, fig)
    for n, iou in enumerate(iou_values):
        i, j = divmod(n, grid_side)
        ax = fig.add_subplot(spec[i, j])
        plot_mc_curve(px, py[iou], ax, names, xlabel=xlabel, ylabel=ylabel, iou=iou_values[n])
    fig.savefig(Path(save_dir), dpi=750)
    plt.close()


def plot_mc_curve(
        px: np.ndarray,
        py: np.ndarray,
        ax: Axes,
        names: list = None,  # noqa: B006
        xlabel: str = "Confidence",
        ylabel: str = "Metric",
        iou: Optional[float] = None
) -> None:
    """Plot metric-confidence curve
    :param px: x axis for plotting
    :param py: y axis for plotting
    :param ax: axis where to plot
    :param names: class names
    :param xlabel: x axis label
    :param ylabel: y axis label
    :param iou: IOU for which the shown plots/metrics belong to
    """
    names = names or []
    if 1 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="blue")  # plot(confidence, metric)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"IOU: {iou:.2f}, best score {py.max():.2f}@{px[py.argmax()]:.3f}")
    ax.grid()
