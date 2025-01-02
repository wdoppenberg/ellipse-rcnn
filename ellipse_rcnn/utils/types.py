from typing import TypedDict, NamedTuple

import torch

from .conics import Ellipse


class TargetDict(TypedDict):
    boxes: torch.Tensor
    labels: torch.Tensor
    image_id: torch.Tensor
    area: torch.Tensor
    iscrowd: torch.Tensor
    ellipse_matrices: torch.Tensor


class PredictionDict(TypedDict):
    bboxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor
    ellipse_matrices: torch.Tensor


type ImageTargetTuple = tuple[torch.Tensor, TargetDict]  # Tensor shape: (C, H, W)
type CollatedBatchType = tuple[
    tuple[torch.Tensor, ...], tuple[TargetDict, ...]
]  # Tensor shape: (C, H, W)
type UncollatedBatchType = list[ImageTargetTuple]

type EllipseType = torch.Tensor | Ellipse


class EllipseTuple(NamedTuple):
    a: float
    b: float
    theta: float
    x: float
    y: float
