import torch
from ellipse_rcnn.utils.conics import ellipse_center


def gaussian_angle_distance(A1: torch.Tensor, A2: torch.Tensor) -> torch.Tensor:
    # Extract covariance matrices (negative of the top-left blocks)
    cov1, cov2 = map(lambda arr: -arr[..., :2, :2], (A1, A2))

    # Extract the means by computing conic centers
    c1_x, c1_y = ellipse_center(A1)
    c2_x, c2_y = ellipse_center(A2)

    # Stack the conic centers into the appropriate shape for computation
    m1 = torch.stack((c1_x, c1_y), dim=-1)[..., None]
    m2 = torch.stack((c2_x, c2_y), dim=-1)[..., None]

    # Compute determinants for covariance matrices
    det_cov1 = torch.clamp(cov1.det(), min=torch.finfo(cov1.dtype).eps)
    det_cov2 = torch.clamp(cov2.det(), min=torch.finfo(cov2.dtype).eps)
    cov_sum = cov1 + cov2

    # Determinant of sum (clamped for numerical stability)
    det_cov_sum = torch.clamp(cov_sum.det(), min=torch.finfo(cov_sum.dtype).eps)

    # Compute fractional term with stabilized determinants
    frac_term = (4 * torch.sqrt(det_cov1 * det_cov2)) / det_cov_sum
    # Stable computation of the exponential term
    mean_diff = m1 - m2
    cov_sum_inv = torch.linalg.solve(
        cov_sum, torch.eye(cov_sum.size(-1), dtype=cov_sum.dtype, device=cov_sum.device)
    )
    exp_arg = -0.5 * mean_diff.transpose(-1, -2) @ cov1 @ cov_sum_inv @ cov2 @ mean_diff
    exp_term = torch.exp(torch.clamp(exp_arg, min=-50, max=50)).squeeze()

    angle_term = frac_term * exp_term

    return torch.arccos(angle_term)


class GaussianAngleDistanceLoss(torch.nn.Module):
    """
    Computes the Gaussian Angle Distance loss between two tensors.

    This class serves as a wrapper around the `gaussian_angle_distance` function,
    providing a clean interface and ensuring numerical stability.

    Attributes
    ----------
    normalize : bool

    nan_to_num : float
        The value to replace NaN entries in the computation with. Helps maintain numerical
        stability in cases where the input tensors contain undefined or invalid values.
    """

    def __init__(self, normalize: bool = True, nan_to_num: float = 10.0):
        super().__init__()
        self.nan_to_num = nan_to_num

    def forward(self, A1: torch.Tensor, A2: torch.Tensor) -> torch.Tensor:
        # Calculate the Gaussian angle distance
        distance = gaussian_angle_distance(A1, A2)

        # Replace NaN values with a predefined constant for numerical stability
        distance = torch.nan_to_num(distance, nan=self.nan_to_num)

        return distance
