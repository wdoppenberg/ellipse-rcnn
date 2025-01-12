from typing import Tuple

import torch
from torch import Tensor
from torchvision.models.detection.transform import (
    GeneralizedRCNNTransform,
    resize_boxes,
    _resize_image_and_masks,
)

from ellipse_rcnn.core.types import PredictionDict, TargetDict


def resize_ellipses(
    ellipses: Tensor, original_size: tuple[int, int], new_size: tuple[int, int]
) -> Tensor:
    """
    Resize ellipses based on image size transformation.

    Parameters
    ----------
    ellipses : Tensor
        [N, 5] tensor of ellipse parameters (a, b, cx, cy, theta).
    original_size : tuple[int, int]
        Original image size [H, W].
    new_size : tuple[int, int]
        New image size [H, W].

    Returns
    -------
    Tensor
        Resized ellipse parameters.
    """
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=ellipses.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=ellipses.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios

    # Unpack parameters
    a, b, cx, cy, theta = ellipses.unbind(1)

    # Scale semi-axes and center coordinates
    # Note: theta remains unchanged as rotation is invariant to scaling
    a = a * ratio_width  # Scale a with width ratio
    b = b * ratio_height  # Scale b with height ratio
    cx = cx * ratio_width  # Scale center x with width ratio
    cy = cy * ratio_height  # Scale center y with height ratio

    return torch.stack((a, b, cx, cy, theta), dim=1)


class EllipseRCNNTransform(GeneralizedRCNNTransform):
    def resize(
        self,
        image: Tensor,
        target: TargetDict | None = None,
    ) -> Tuple[Tensor, TargetDict | None]:
        h, w = image.shape[-2:]
        if self.training:
            if self._skip_resize:
                return image, target
            size = self.torch_choice(self.min_size)  # type: ignore
        else:
            size = self.min_size[-1]  # type: ignore
        image, target = _resize_image_and_masks(
            image, size, self.max_size, target, self.fixed_size
        )

        if target is None:
            return image, target

        # Handle bounding boxes
        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])  # type: ignore
        target["boxes"] = bbox

        # Handle ellipse parameters
        ellipses = target["ellipse_params"]
        ellipses = resize_ellipses(ellipses, (h, w), image.shape[-2:])  # type: ignore
        target["ellipse_params"] = ellipses

        return image, target

    def postprocess(
        self,
        result: list[PredictionDict],
        image_shapes: list[tuple[int, int]],
        original_image_sizes: list[tuple[int, int]],
    ) -> list[PredictionDict]:
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(
            zip(result, image_shapes, original_image_sizes)
        ):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)  # type: ignore
            result[i]["boxes"] = boxes

            ellipses = pred["ellipse_params"]
            ellipses = resize_ellipses(ellipses, im_s, o_im_s)
            result[i]["ellipse_params"] = ellipses

        return result
