from typing import List, Tuple, Union, Iterable, TypedDict

import torch
from utils.conics import Ellipse


class TargetDict(TypedDict):
    bboxes: torch.Tensor
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


ImageTargetTuple = Tuple[torch.Tensor, TargetDict]
CollatedBatchType = Tuple[torch.Tensor, List[TargetDict]]
RawBatchType = Iterable[CollatedBatchType]

EllipseType = Union[torch.Tensor, Ellipse]
