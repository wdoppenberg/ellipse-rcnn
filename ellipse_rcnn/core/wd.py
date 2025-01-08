import torch

from ellipse_rcnn.core.ops import ellipse_center


def wasserstein_distance(
    A1: torch.Tensor,
    A2: torch.Tensor,
    *,
    shape_only: bool = False,
) -> torch.Tensor:
    """
    Compute the squared Wasserstein-2 distance between ellipses represented by their matrices.

    Args:
        A1, A2: Ellipse matrices of shape (..., 3, 3)
        shape_only: If True, ignores displacement term

    Returns:
        Tensor containing Wasserstein distances
    """
    # Ensure batch sizes match
    if A1.shape[:-2] != A2.shape[:-2]:
        raise ValueError(
            f"Batch size mismatch: A1 has shape {A1.shape[:-2]}, A2 has shape {A2.shape[:-2]}"
        )

    # Extract covariance matrices (upper 2x2 blocks)
    cov1 = A1[..., :2, :2]
    cov2 = A2[..., :2, :2]

    if shape_only:
        displacement_term = 0
    else:
        # Compute centers
        m1 = torch.vstack(ellipse_center(A1)).T[..., None]
        m2 = torch.vstack(ellipse_center(A2)).T[..., None]

        # Mean difference term
        displacement_term = torch.sum((m1 - m2) ** 2, dim=(1, 2))

    # Compute the matrix square root term
    eigenvalues1, eigenvectors1 = torch.linalg.eigh(cov1)
    sqrt_eigenvalues1 = torch.sqrt(torch.clamp(eigenvalues1, min=1e-7))
    sqrt_cov1 = (
        eigenvectors1
        @ torch.diag_embed(sqrt_eigenvalues1)
        @ eigenvectors1.transpose(-2, -1)
    )

    inner_term = sqrt_cov1 @ cov2 @ sqrt_cov1
    eigenvalues_inner, eigenvectors_inner = torch.linalg.eigh(inner_term)
    sqrt_inner = (
        eigenvectors_inner
        @ torch.diag_embed(torch.sqrt(torch.clamp(eigenvalues_inner, min=1e-7)))
        @ eigenvectors_inner.transpose(-2, -1)
    )

    trace_term = (
        torch.diagonal(cov1, dim1=-2, dim2=-1).sum(-1)
        + torch.diagonal(cov2, dim1=-2, dim2=-1).sum(-1)
        - 2 * torch.diagonal(sqrt_inner, dim1=-2, dim2=-1).sum(-1)
    )

    return displacement_term + trace_term


def symmetric_wasserstein_distance(
    A1: torch.Tensor,
    A2: torch.Tensor,
    *,
    shape_only: bool = False,
    nan_to_num: float = float(1e4),
    normalize: bool = False,
) -> torch.Tensor:
    """
    Compute symmetric Wasserstein distance between ellipses.

    Args:
        A1, A2: Ellipse matrices
        shape_only: If True, ignores displacement term
        nan_to_num: Value to replace NaN entries with
        normalize: If True, normalizes the output to [0, 1]
    """
    w = torch.nan_to_num(
        wasserstein_distance(A1, A2, shape_only=shape_only), nan=nan_to_num
    )

    if w.lt(0).any():
        raise ValueError("Negative Wasserstein distance encountered.")

    if normalize:
        w = 1 - torch.exp(-w)
    return w


class WassersteinLoss(torch.nn.Module):
    """
    Computes the Wasserstein distance loss between two ellipse tensors.

    The Wasserstein distance provides a natural metric for comparing probability
    distributions or shapes, with advantages over KL divergence such as:
    - It's symmetric by definition
    - It provides a true metric (satisfies triangle inequality)
    - It's well-behaved even when distributions have different supports

    Attributes:
        shape_only: If True, computes distance based on shape without considering position
        nan_to_num: Value to replace NaN entries with
        normalize: If True, normalizes output to [0, 1] using exponential scaling
    """

    def __init__(
        self, shape_only: bool = True, nan_to_num: float = 10.0, normalize: bool = False
    ):
        super().__init__()
        self.shape_only = shape_only
        self.nan_to_num = nan_to_num
        self.normalize = normalize

    def forward(self, A1: torch.Tensor, A2: torch.Tensor) -> torch.Tensor:
        return symmetric_wasserstein_distance(
            A1,
            A2,
            shape_only=self.shape_only,
            nan_to_num=self.nan_to_num,
            normalize=self.normalize,
        )
