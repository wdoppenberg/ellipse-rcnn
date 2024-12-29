import torch

from ellipse_rcnn.utils.conics import conic_center


def mv_kullback_leibler_divergence(
    A1: torch.Tensor,
    A2: torch.Tensor,
    *,
    shape_only: bool = False,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """
    Compute KL divergence between ellipses represented by their matrices.

    Args:
        A1, A2: Ellipse matrices of shape (..., 3, 3)
        shape_only: If True, ignores displacement term
        epsilon: Small value for numerical stability
    """
    # Extract the upper 2x2 blocks as covariance matrices
    cov1 = A1[..., :2, :2]
    cov2 = A2[..., :2, :2]

    # Add small epsilon to diagonal for stability and ensure numerical robustness
    eye = torch.eye(2, device=A1.device)
    cov1 = cov1 + eye * epsilon
    cov2 = cov2 + eye * epsilon

    # Compute centers
    m1 = torch.vstack(conic_center(A1)).T[..., None]
    m2 = torch.vstack(conic_center(A2)).T[..., None]

    # Compute inverse
    try:
        cov2_inv = torch.linalg.inv(cov2)
    except RuntimeError:
        cov2_inv = torch.linalg.pinv(cov2)

    # Trace term
    trace_term = (cov2_inv @ cov1).diagonal(dim2=-2, dim1=-1).sum(1)

    # Log determinant term
    # Clamp determinants to avoid instability
    det_cov1 = torch.clamp(torch.det(cov1), min=epsilon)
    det_cov2 = torch.clamp(torch.det(cov2), min=epsilon)
    log_term = torch.log(det_cov2 / det_cov1)

    if shape_only:
        displacement_term = 0
    else:
        # Mean difference term
        displacement_term = torch.clamp(
            (m1 - m2).transpose(-1, -2) @ cov2_inv @ (m1 - m2), min=epsilon, max=1e5
        ).squeeze()

    return 0.5 * (trace_term + displacement_term - 2 + log_term)


def symmetric_kl_divergence(
    A1: torch.Tensor,
    A2: torch.Tensor,
    *,
    shape_only: bool = True,
    nan_to_num: float = 10.0,
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

    if normalize:
        kl = 1 - torch.exp(-kl)
    return kl


class SymmetricKLDLoss(torch.nn.Module):
    """
    Computes the symmetric Kullback-Leibler divergence (KLD) loss between two tensors.

    SymmetricKLDLoss is used for measuring the divergence between two probability
    distributions or tensors, which can be useful in tasks such as generative modeling
    or optimization. The function allows for options such as normalizing the tensors or
    focusing only on their shape for comparison. Additionally, it includes a feature
    to handle NaN values by replacing them with a numeric constant.

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
