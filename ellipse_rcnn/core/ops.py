import torch


def ellipse_to_conic_matrix(
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
    theta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Converts parametric ellipse attributes into a general conic matrix
    representation in 2D Cartesian space.
    _[1]:

      | A = a²(sin θ)² + b²(cos θ)²
      | B = 2(b² - a²) sin θ cos θ
      | C = a²(cos θ)² + b²(sin θ)²
      | D = -2Ax₀ - By₀
      | E = -Bx₀ - 2Cy₀
      | F = Ax₀² + Bx₀y₀ + Cy₀² - a²b²

    Parameters:
        a: torch.Tensor
            Semi-major axis length.
        b: torch.Tensor
            Semi-minor axis length.
        x: torch.Tensor, optional
            X-coordinate of the center. Defaults to 0.
        y: torch.Tensor, optional
            Y-coordinate of the center. Defaults to 0.
        theta: torch.Tensor, optional
            Rotation angle w.r.t. x-axis in radians. Defaults to 0.

    Returns:
        torch.Tensor of shape [N, 3, 3]
            A general conic matrix of the ellipse with parameters encoded.

    Returns
    -------
    torch.Tensor
        Array of ellipse matrices

    References
    ----------
    .. [1] https://www.researchgate.net/publication/355490899_Lunar_Crater_Identification_in_Digital_Images
    """

    x = x if x is not None else torch.zeros_like(a)
    y = y if y is not None else torch.zeros_like(a)
    theta = theta if theta is not None else torch.zeros_like(a)

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    a2 = a**2
    b2 = b**2

    A = a2 * sin_theta**2 + b2 * cos_theta**2
    B = 2 * (b2 - a2) * sin_theta * cos_theta
    C = a2 * cos_theta**2 + b2 * sin_theta**2
    D = -2 * A * x - B * y
    F = -B * x - 2 * C * y
    G = A * (x**2) + B * x * y + C * (y**2) - a2 * b2

    # Create (array of) of conic matrix (N, 3, 3)
    conic_matrix = torch.stack(
        tensors=(
            torch.stack((A, B / 2, D / 2), dim=-1),
            torch.stack((B / 2, C, F / 2), dim=-1),
            torch.stack((D / 2, F / 2, G), dim=-1),
        ),
        dim=-1,
    )

    return conic_matrix.squeeze()


@torch.jit.script
def ellipse_center(conic_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns center of ellipse in 2D cartesian coordinate system with numerical stability."""
    # Extract the top-left 2x2 submatrix of the conic matrix
    A = conic_matrix[..., :2, :2]

    A_inv = torch.linalg.inv(A)

    # Extract the last two rows for the linear term
    b = -conic_matrix[..., :2, 2][..., None]

    # Stabilize any potential numerical instabilities
    centers = torch.matmul(A_inv, b).squeeze()

    cx, cy = centers[..., 0].reshape(-1), centers[..., 1].reshape(-1)

    return cx, cy


@torch.jit.script
def ellipse_axes(conic_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns semi-major and semi-minor axes of ellipse in 2D cartesian coordinate system."""
    lambdas = (
        torch.linalg.eigvalsh(conic_matrix[..., :2, :2])
        / (-torch.det(conic_matrix) / torch.det(conic_matrix[..., :2, :2]))[..., None]
    )
    axes = torch.sqrt(1 / lambdas)
    return axes[..., 0].reshape(-1), axes[..., 1].reshape(-1)


@torch.jit.script
def ellipse_angle(conic_matrix: torch.Tensor) -> torch.Tensor:
    """Returns angle of ellipse in radians w.r.t. x-axis."""
    return (
        -torch.atan2(
            2 * conic_matrix[..., 1, 0],
            conic_matrix[..., 1, 1] - conic_matrix[..., 0, 0],
        )
        / 2
    ).reshape(-1)


def bbox_ellipse(ellipses: torch.Tensor) -> torch.Tensor:
    """Converts ellipse parameters to bounding box tensors with format [xmin, ymin, xmax, ymax].

    Parameters
    ----------
    ellipses : torch.Tensor
        Array of ellipse parameters with shape [N, 5] with ordering [a, b, cx, cy, theta]

    Returns
    -------
        Array of bounding boxes
    """
    a, b, cx, cy, theta = ellipses.unbind(-1)
    ux, uy = a * torch.cos(theta), a * torch.sin(theta)
    vx, vy = (
        b * torch.cos(theta + torch.pi / 2),
        b * torch.sin(theta + torch.pi / 2),
    )

    box_halfwidth = torch.sqrt(ux**2 + vx**2)
    box_halfheight = torch.sqrt(uy**2 + vy**2)

    bboxes = torch.vstack(
        (
            cx - box_halfwidth,
            cy - box_halfheight,
            cx + box_halfwidth,
            cy + box_halfheight,
        )
    ).T

    return bboxes


@torch.jit.script
def bbox_ellipse_matrix(
    ellipses: torch.Tensor,
) -> torch.Tensor:
    """Converts (array of) ellipse matrices to bounding box tensor with format [xmin, ymin, xmax, ymax].

    Parameters
    ----------
    ellipses:
        Array of ellipse matrices

    Returns
    -------
        Array of bounding boxes
    """
    cx, cy = ellipse_center(ellipses)
    theta = ellipse_angle(ellipses)
    a, b = ellipse_axes(ellipses)

    ellipses_p = torch.stack([a, b, cx, cy, theta]).reshape(-1, 5)

    return bbox_ellipse(ellipses_p)


def ellipse_area(ellipses: torch.Tensor) -> torch.Tensor:
    """Calculates the area of the given ellipses.

    Parameters
    ----------
    ellipses : torch.Tensor
        Array of ellipse parameters with shape [N, 5] with ordering [a, b, cx, cy, theta]

    Returns
    -------
        Array of bounding boxes
    """
    a, b = ellipses[:, 0], ellipses[:, 1]
    return a * b * torch.pi


def remove_small_ellipses(ellipses: torch.Tensor, min_size: float) -> torch.Tensor:
    """
    Remove every ellipse from `ellipses` where either axis is smaller than `min_size`.

    Parameters
    ----------
    ellipses : torch.Tensor
        Array of ellipses with shape `[N, 5]` in the format `[a, b, cx, cy, theta]`:
        - `a` (float): Semi-major axis.
        - `b` (float): Semi-minor axis.
        - `cx` (float): x-coordinate of the center.
        - `cy` (float): y-coordinate of the center.
        - `theta` (float): Rotation angle (in radians).
    min_size : float
        Minimum size required for each axis (`a` and `b`) of the ellipses.

    Returns
    -------
    torch.Tensor
        Indices of ellipses that have both axes larger than or equal to `min_size`.

    Examples
    --------
    >>> ellipses = torch.tensor([[3.0, 2.0, 0.0, 0.0, 0.0],
    ...                          [1.0, 1.5, 1.0, 1.0, 0.1],
    ...                          [4.0, 3.0, 2.0, 3.0, 0.5]])
    >>> min_size = 2.0
    >>> remove_small_ellipses(ellipses, min_size)
    tensor([0, 2])
    """
    a, b = ellipses[:, 0], ellipses[:, 1]
    keep = (a >= min_size) & (b >= min_size)
    return torch.where(keep)[0]
