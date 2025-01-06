import torch

from ellipse_rcnn.utils.conics import ellipse_to_conic_matrix


def sample_parametric_ellipses(
    batch_size: int,
    a_range=(2.0, 5.0),
    b_range=(1.0, 3.0),
    theta_range=(0.0, 2 * torch.pi),
    xy_range=(-1.0, 1.0),
) -> tuple[torch.Tensor, ...]:
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

    return a, b, x, y, theta


def sample_conic_ellipses(
    batch_size: int,
    a_range=(2.0, 5.0),
    b_range=(1.0, 3.0),
    theta_range=(0.0, 2 * torch.pi),
    xy_range=(-1.0, 1.0),
) -> torch.Tensor:
    a, b, x, y, theta = sample_parametric_ellipses(
        batch_size, a_range, b_range, theta_range, xy_range
    )
    return ellipse_to_conic_matrix(a=a, b=b, x=x, y=y, theta=theta)
