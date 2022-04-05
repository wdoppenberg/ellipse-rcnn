from typing import TypedDict, Union
import torch

from utils.conics import Ellipse


class TargetDict(TypedDict):
    bboxes: torch.Tensor
    labels: torch.Tensor
    image_id: torch.Tensor
    area: torch.Tensor
    iscrowd: torch.Tensor
    ellipse_matrices: torch.Tensor


EllipseType = Union[torch.Tensor, Ellipse]
