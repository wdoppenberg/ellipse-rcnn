import torch

from ellipse_rcnn.core.ops import ellipse_center


def mv_kullback_leibler_divergence(
    A1: torch.Tensor,
    A2: torch.Tensor,
    *,
    shape_only: bool = False,
) -> torch.Tensor:
    """
    Compute multi-variate KL divergence between ellipses represented by their matrices.

    Parameters
    ----------
    A1, A2: Ellipse matrices of shape (..., 3, 3)
    shape_only: If True, ignores displacement term
    """

    # Ensure that batch sizes are equal
    if A1.shape[:-2] != A2.shape[:-2]:
        raise ValueError(
            f"Batch size mismatch: A1 has shape {A1.shape[:-2]}, A2 has shape {A2.shape[:-2]}"
        )

    # Extract the upper 2x2 blocks as covariance matrices
    cov1 = A1[..., :2, :2]
    cov2 = A2[..., :2, :2]

    # Compute centers
    m1 = torch.vstack(ellipse_center(A1)).T[..., None]
    m2 = torch.vstack(ellipse_center(A2)).T[..., None]

    # Compute inverse
    try:
        cov2_inv = torch.linalg.inv(cov2)
    except RuntimeError:
        cov2_inv = torch.linalg.pinv(cov2)

    # Trace term
    trace_term = (cov2_inv @ cov1).diagonal(dim2=-2, dim1=-1).sum(1)

    # Log determinant term
    det_cov1 = torch.det(cov1)
    det_cov2 = torch.det(cov2)
    log_term = torch.log(det_cov2 / det_cov1).nan_to_num(nan=0.0)

    if shape_only:
        displacement_term = 0
    else:
        # Mean difference term
        displacement_term = (
            ((m1 - m2).transpose(-1, -2) @ cov2_inv @ (m1 - m2)).squeeze().abs()
        )

    return 0.5 * (trace_term + displacement_term - 2 + log_term)


def symmetric_kl_divergence(
    A1: torch.Tensor,
    A2: torch.Tensor,
    *,
    shape_only: bool = False,
    nan_to_num: float = float(1e4),
    normalize: bool = False,
) -> torch.Tensor:
    """
    Compute symmetric KL divergence between ellipses.
    """
    kl_12 = torch.nan_to_num(
        mv_kullback_leibler_divergence(A1, A2, shape_only=shape_only), nan_to_num
    )
    kl_21 = torch.nan_to_num(
        mv_kullback_leibler_divergence(A2, A1, shape_only=shape_only), nan_to_num
    )
    kl = (kl_12 + kl_21) / 2

    if kl.lt(0).any():
        raise ValueError("Negative KL divergence encountered.")

    if normalize:
        kl = 1 - torch.exp(-kl)
    return kl


class SymmetricKLDLoss(torch.nn.Module):
    """
    Computes the symmetric Kullback-Leibler divergence (KLD) loss between two tensors.

    Attributes
    ----------
    shape_only : bool
        If True, computes the divergence based on the shape of the tensors only. This
        can be used to evaluate similarity without considering magnitude differences.
    nan_to_num : float
        The value to replace NaN entries in the tensors with. Helps maintain numerical
        stability in cases where the input tensors contain undefined or invalid values.
    normalize : bool
        If True, normalizes the tensors before computing the divergence. This is
        typically used when the inputs are not already probability distributions.
    """

    def __init__(
        self, shape_only: bool = True, nan_to_num: float = 10.0, normalize: bool = False
    ):
        super().__init__()
        self.shape_only = shape_only
        self.nan_to_num = nan_to_num
        self.normalize = normalize

    def forward(self, A1: torch.Tensor, A2: torch.Tensor) -> torch.Tensor:
        return symmetric_kl_divergence(
            A1,
            A2,
            shape_only=self.shape_only,
            nan_to_num=self.nan_to_num,
            normalize=self.normalize,
        )
