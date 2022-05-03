from typing import Union

import numpy as np
import torch
import numpy.linalg as LA

from ..utils.conics import conic_center, unimodular_matrix


def mv_kullback_leibler_divergence(A1: torch.Tensor, A2: torch.Tensor, shape_only: bool = False) -> torch.Tensor:
    A1, A2 = map(unimodular_matrix, (A1, A2))
    cov1, cov2 = map(lambda arr: -arr[..., :2, :2], (A1, A2))
    m1, m2 = map(lambda arr: torch.vstack(conic_center(arr)).T[..., None], (A1, A2))

    trace_term = (torch.inverse(cov1) @ cov2).diagonal(dim2=-2, dim1=-1).sum(1)
    log_term = torch.log(torch.det(cov1) / torch.det(cov2))

    if shape_only:
        displacement_term = 0
    else:
        displacement_term = ((m1 - m2).transpose(-1, -2) @ cov1.inverse() @ (m1 - m2)).squeeze()

    return 0.5 * (trace_term + displacement_term - 2 + log_term)


def norm_mv_kullback_leibler_divergence(A1: torch.Tensor, A2: torch.Tensor) -> torch.Tensor:
    return 1 - torch.exp(-mv_kullback_leibler_divergence(A1, A2))


def gaussian_angle_distance(A1: Union[torch.Tensor, np.ndarray], A2: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    A1, A2 = map(unimodular_matrix, (A1, A2))
    cov1, cov2 = map(lambda arr: -arr[..., :2, :2], (A1, A2))

    if isinstance(cov1, torch.Tensor) and isinstance(cov2, torch.Tensor):
        m1, m2 = map(lambda arr: torch.vstack(tuple(conic_center(arr))).T[..., None], (A1, A2))

        frac_term = (4 * torch.sqrt(cov1.det() * cov2.det())) / (cov1 + cov2).det()
        exp_term = torch.exp(
            -0.5 * (m1 - m2).transpose(-1, -2) @ cov1 @ (cov1 + cov2).inverse() @ cov2 @ (m1 - m2)
        ).squeeze()

        return (frac_term * exp_term).arccos()

    elif isinstance(cov1, np.ndarray) and isinstance(cov2, np.ndarray):
        m1, m2 = map(lambda arr: np.vstack(conic_center(arr)).T[..., None], (A1, A2))

        frac_term = 4 * np.sqrt(LA.det(cov1) * LA.det(cov2)) / (LA.det(cov1 + cov2))
        exp_term = np.exp(-0.5 * (m1 - m2).transpose(0, 2, 1) @ cov1 @ LA.inv(cov1 + cov2) @ cov2 @ (m1 - m2)).squeeze()

        return np.arccos(frac_term * exp_term)
    else:
        raise TypeError("A1 and A2 should of type torch.Tensor or np.ndarray.")
