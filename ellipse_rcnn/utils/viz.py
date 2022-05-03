from __future__ import annotations

from typing import Dict, Optional

import torch
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.collections import PatchCollection

from ellipse_rcnn.utils.conics import bbox_ellipse, ellipse_angle


class DetectionPlotter:
    def __init__(
        self,
        x_min: torch.Tensor,
        y_min: torch.Tensor,
        x_max: torch.Tensor,
        y_max: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
    ):
        """
        Constructs a DetectionPlotter from a tensor of bounding boxes.

        Parameters
        ----------
        x_min:
            A tensor of the minimum x-coordinates of the bounding boxes.
        y_min:
            A tensor of the minimum y-coordinates of the bounding boxes.
        x_max:
            A tensor of the maximum x-coordinates of the bounding boxes.
        y_max:
            A tensor of the maximum y-coordinates of the bounding boxes.
        theta:
            (Optional) A tensor containing the angle of the ellipses in radians.
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        if theta is None:
            self.theta = torch.zeros_like(x_min)
        else:
            self.theta = theta

    @property
    def w(self) -> torch.Tensor:
        """
        Returns the width of the bounding box.
        """
        return self.x_max - self.x_min

    @property
    def h(self) -> torch.Tensor:
        """
        Returns the height of the bounding box.
        """
        return self.y_max - self.y_min

    @property
    def cx(self) -> torch.Tensor:
        """
        Returns the center x-coordinate of the bounding box.
        """
        return self.x_min + (self.w / 2)

    @property
    def cy(self) -> torch.Tensor:
        """
        Returns the center y-coordinate of the bounding box.
        """
        return self.y_min + (self.h / 2)

    @classmethod
    def from_boxes(cls, boxes: torch.Tensor, theta: Optional[torch.Tensor] = None) -> DetectionPlotter:
        """
        Constructs a DetectionPlotter from a tensor of bounding boxes.

        Parameters
        ----------
        boxes:
            A tensor of bounding boxes in the format [x_min, y_min, x_max, y_max].
        theta:
            (Optional) A tensor of the same shape as `boxes` containing the angle of the ellipses in radians.

        Returns
        -------
        DetectionPlotter
            A DetectionPlotter object.
        """
        return cls(boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3], theta)

    @classmethod
    def from_ellipses(cls, ellipse_matrices: torch.Tensor) -> DetectionPlotter:
        """
        Constructs a DetectionPlotter from a tensor of ellipse matrices.

        Parameters
        ----------
        ellipse_matrices:
            A tensor of ellipse matrices [N, 3, 3].

        Returns
        -------
        DetectionPlotter
            A DetectionPlotter object.
        """
        theta = ellipse_angle(ellipse_matrices)
        return cls.from_boxes(bbox_ellipse(ellipse_matrices), theta)

    def plot(
        self,
        ax: Axes,
        convert_to_degrees: bool = True,
        ellipse_kwargs: Optional[Dict] = None,
        rectangle_kwargs: Optional[Dict] = None,
    ) -> Axes:
        """
        Plots the bounding boxes and ellipses on the given axes.

        Parameters
        ----------
        ax:
            The axes to plot the bounding boxes and ellipses on.
        convert_to_degrees:
            Whether to convert the angles to degrees.
        ellipse_kwargs:
            (Optional) A dictionary of keyword arguments to pass to each Ellipse initializer.
        rectangle_kwargs:
            (Optional) A dictionary of keyword arguments to pass to each Rectangle initializer.

        Returns
        -------
        Axes:
            The axes with the bounding boxes and ellipses plotted.
        """
        if ellipse_kwargs is None:
            ellipse_kwargs = dict(color="b", alpha=1, fill=False)

        if rectangle_kwargs is None:
            rectangle_kwargs = dict(color="r", alpha=1, fill=False)

        if convert_to_degrees:
            theta = torch.rad2deg(self.theta)
        else:
            theta = self.theta

        ellipses = [
            Ellipse((cx, cy), w, h, t, **ellipse_kwargs)
            for cx, cy, w, h, t in zip(self.cx, self.cy, self.w, self.h, theta)
        ]
        ax.add_collection(PatchCollection(ellipses, match_original=True))

        rectangles = [
            Rectangle((x_min, y_min), w, h, **rectangle_kwargs)
            for x_min, y_min, w, h in zip(self.x_min, self.y_min, self.w, self.h)
        ]
        ax.add_collection(PatchCollection(rectangles, match_original=True))

        return ax
