from typing import Literal

import numpy as np
import torch
from torchvision.ops import boxes as box_ops
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import EllipseCollection, PatchCollection
from matplotlib.patches import Rectangle
from ellipse_rcnn.core.ops import ellipse_center


def plot_ellipses(
    ellipse_params: torch.Tensor,
    figsize: tuple[float, float] = (15, 15),
    plot_centers: bool = False,
    ax: Axes | None = None,
    rim_color: str = "r",
    alpha: float = 1.0,
) -> None:
    a, b, cx, cy, theta = ellipse_params.unbind(-1)

    a, b, theta, cx, cy = map(
        lambda t: t.detach().cpu().numpy(),
        (a, b, theta, cx, cy),
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"aspect": "equal"})

    ec = EllipseCollection(
        a * 2,
        b * 2,
        np.degrees(theta),
        units="xy",
        offsets=np.column_stack((cx, cy)),
        transOffset=ax.transData,
        facecolors="None",
        edgecolors=rim_color,
        linewidths=2.5,
        alpha=alpha,
    )
    ax.add_collection(ec)

    if plot_centers:
        crater_centers = ellipse_center(ellipse_params)
        for k, c_i in enumerate(crater_centers):
            x, y = c_i[0], c_i[1]
            ax.text(x.item(), y.item(), str(k), color=rim_color)


def plot_bboxes(
    boxes: torch.Tensor,
    box_type: Literal["xyxy", "xywh", "cxcywh"] = "xyxy",
    figsize: tuple[float, float] = (15, 15),
    plot_centers: bool = False,
    ax: Axes | None = None,
    rim_color: str = "r",
    alpha: float = 1.0,
) -> None:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"aspect": "equal"})

    if box_type != "xyxy":
        boxes = box_ops.box_convert(boxes, box_type, "xyxy")

    boxes = boxes.detach().cpu().numpy()
    rectangles = []
    for k, b_i in enumerate(boxes):
        x1, y1, x2, y2 = b_i
        rectangles.append(Rectangle((x1, y1), x2 - x1, y2 - y1))

    collection = PatchCollection(
        rectangles, edgecolor=rim_color, facecolor="none", alpha=alpha
    )
    ax.add_collection(collection)

    if plot_centers:
        for k, b_i in enumerate(boxes):
            x1, y1, x2, y2 = b_i
            ax.text(x1, y1, str(k), color=rim_color)
