from typing import TypedDict, NamedTuple

import torch


class TargetDict(TypedDict):
    boxes: torch.Tensor
    labels: torch.Tensor
    image_id: torch.Tensor
    area: torch.Tensor
    iscrowd: torch.Tensor
    ellipse_params: torch.Tensor


class LossDict(TypedDict, total=False):
    loss_classifier: torch.Tensor
    loss_box_reg: torch.Tensor
    loss_objectness: torch.Tensor
    loss_rpn_box_reg: torch.Tensor
    loss_ellipse_kld: torch.Tensor
    loss_ellipse_smooth_l1: torch.Tensor
    loss_total: torch.Tensor


class PredictionDict(TypedDict):
    bboxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor
    ellipse_params: torch.Tensor


type ImageTargetTuple = tuple[torch.Tensor, TargetDict]  # Tensor shape: (C, H, W)
type CollatedBatchType = tuple[
    tuple[torch.Tensor, ...], tuple[TargetDict, ...]
]  # Tensor shape: (C, H, W)
type UncollatedBatchType = list[ImageTargetTuple]

type EllipseType = torch.Tensor


class EllipseTuple(NamedTuple):
    a: float
    b: float
    theta: float
    x: float
    y: float
