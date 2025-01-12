import torch

from ellipse_rcnn.core.ops import ellipse_to_conic_matrix


def sample_parametric_ellipses(
    batch_size: int,
    a_range=(2.0, 5.0),
    b_range=(1.0, 3.0),
    theta_range=(0.0, 2 * torch.pi),
    xy_range=(-1.0, 1.0),
) -> torch.Tensor:
    """
    Generate parametric ellipses with uniformly sampled random parameters.

    Parameters:
        batch_size (int): Number of ellipses to generate parameters for.
        a_range (tuple[float, float]): Range of values for the semi-major axis 'a'.
        b_range (tuple[float, float]): Range of values for the semi-minor axis 'b'.
        theta_range (tuple[float, float]): Range of values for the rotational angle 'theta' in radians.
        xy_range (tuple[float, float]): Range of values for the x and y positions of the ellipse center.

    Returns:
        tuple[torch.Tensor, ...]: A tuple containing five tensors:
            - `a`: Tensor containing semi-major axes of shape (batch_size,).
            - `b`: Tensor containing semi-minor axes of shape (batch_size,).
            - `x`: Tensor containing x positions of shape (batch_size,).
            - `y`: Tensor containing y positions of shape (batch_size,).
            - `theta`: Tensor containing angle in radians of shape (batch_size,).
    """
    # Semi-major and semi-minor axes
    a = torch.rand(batch_size) * (a_range[1] - a_range[0]) + a_range[0]
    b = torch.rand(batch_size) * (b_range[1] - b_range[0]) + b_range[0]
    # Ensure b <= a
    b = torch.minimum(a, b)

    # Angle in radians
    theta = torch.rand(batch_size) * (theta_range[1] - theta_range[0]) + theta_range[0]

    # Positions
    x = torch.rand(batch_size) * (xy_range[1] - xy_range[0]) + xy_range[0]
    y = torch.rand(batch_size) * (xy_range[1] - xy_range[0]) + xy_range[0]

    return torch.stack([a, b, x, y, theta], dim=-1).view(-1, 5)


def sample_conic_ellipses(
    batch_size: int,
    a_range=(2.0, 5.0),
    b_range=(1.0, 3.0),
    theta_range=(0.0, 2 * torch.pi),
    xy_range=(-1.0, 1.0),
) -> torch.Tensor:
    a, b, x, y, theta = sample_parametric_ellipses(
        batch_size, a_range, b_range, theta_range, xy_range
    ).unbind(-1)
    return ellipse_to_conic_matrix(a=a, b=b, x=x, y=y, theta=theta)
