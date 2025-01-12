from typing import TypedDict, NamedTuple

from torch import Tensor


class TargetDict(TypedDict):
    boxes: Tensor  # Tensor containing bounding boxes, shape (N, 4) [x1, y1, x2, y2]
    labels: Tensor  # Tensor of class labels for each object, shape (N,)
    image_id: Tensor  # Tensor with the image ID, scalar tensor
    area: Tensor  # Tensor containing the area of each bounding box, shape (N,)
    iscrowd: Tensor  # Tensor indicating if an object is a crowd, shape (N,)
    ellipse_params: Tensor  # Tensor containing ellipse parameters, shape (N, 5) [a, b, cx, cy, theta]


class LossDict(TypedDict, total=False):
    loss_classifier: Tensor
    loss_ellipse_reg: Tensor
    loss_objectness: Tensor
    loss_rpn_box_reg: Tensor
    loss_total: Tensor


class PredictionDict(TypedDict):
    boxes: Tensor
    labels: Tensor
    scores: Tensor
    ellipse_params: Tensor


type ImageTargetTuple = tuple[Tensor, TargetDict]  # Tensor shape: (C, H, W)
type CollatedBatchType = tuple[
    tuple[Tensor, ...], tuple[TargetDict, ...]
]  # Tensor shape: (C, H, W)
type UncollatedBatchType = list[ImageTargetTuple]


class EllipseTuple(NamedTuple):
    a: float
    b: float
    x: float
    y: float
    theta: float
