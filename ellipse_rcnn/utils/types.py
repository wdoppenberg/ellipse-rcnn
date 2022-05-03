from typing import List, Tuple, Union, TypedDict

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


ImageTargetTuple = Tuple[torch.Tensor, TargetDict]  # Tensor shape: (C, H, W)
CollatedBatchType = Tuple[Tuple[torch.Tensor, ...], Tuple[TargetDict, ...]]  # Tensor shape: (C, H, W)
UncollatedBatchType = List[ImageTargetTuple]

EllipseType = Union[torch.Tensor, Ellipse]
