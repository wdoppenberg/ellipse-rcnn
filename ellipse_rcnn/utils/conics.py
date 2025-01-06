from typing import Literal

import torch


@torch.jit.script
def adjugate_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Return adjugate matrix [1].

    Parameters
    ----------
    matrix:
        Input matrix

    Returns
    -------
    torch.Tensor
        Adjugate of input matrix

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Adjugate_matrix
    """

    cofactor = torch.inverse(matrix).T * torch.det(matrix)
    return cofactor.T


# @torch.jit.script
def unimodular_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Rescale matrix such that det(ellipses) = 1, in other words, make it unimodular. Doest not work with tensors
    of dtype torch.float64.

    Parameters
    ----------
    matrix:
        Matrix input

    Returns
    -------
    torch.Tensor
        Unimodular version of input matrix.
    """
    val = 1.0 / torch.det(matrix)
    return (torch.sign(val) * torch.pow(torch.abs(val), 1.0 / 3.0))[
        ..., None, None
    ] * matrix


# @torch.jit.script
def ellipse_to_conic_matrix(
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
    theta: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Returns matrix representation for crater derived from ellipse parameters such that _[1]:

      | A = a²(sin θ)² + b²(cos θ)²
      | B = 2(b² - a²) sin θ cos θ
      | C = a²(cos θ)² + b²(sin θ)²
      | D = -2Ax₀ - By₀
      | E = -Bx₀ - 2Cy₀
      | F = Ax₀² + Bx₀y₀ + Cy₀² - a²b²

    Resulting in a conic matrix:
    ::
                |A    B/2  D/2 |
        M  =    |B/2  C    E/2 |
                |D/2  E/2  G   |

    Parameters
    ----------
    a:
        Semi-Major ellipse axis
    b:
        Semi-Minor ellipse axis
    theta:
        Ellipse angle (radians)
    x:
        X-position in 2D cartesian coordinate system (coplanar)
    y:
        Y-position in 2D cartesian coordinate system (coplanar)

    Returns
    -------
    torch.Tensor
        Array of ellipse matrices

    References
    ----------
    .. [1] https://www.researchgate.net/publication/355490899_Lunar_Crater_Identification_in_Digital_Images
    """

    x = x if x is not None else torch.zeros(1)
    y = y if y is not None else torch.zeros(1)
    theta = theta if theta is not None else torch.zeros(1)

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


def conic_center(conic_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns center of ellipse in 2D cartesian coordinate system with numerical stability."""
    # Extract the top-left 2x2 submatrix of the conic matrix
    A = conic_matrix[..., :2, :2]

    # Add stabilization for pseudoinverse computation by clamping singular values
    A_pinv = torch.linalg.pinv(A, rcond=torch.finfo(A.dtype).eps)

    # Extract the last two rows for the linear term
    b = -conic_matrix[..., :2, 2][..., None]

    # Stabilize any potential numerical instabilities
    centers = torch.matmul(A_pinv, b).squeeze()

    return centers[..., 0], centers[..., 1]


def ellipse_axes(conic_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns semi-major and semi-minor axes of ellipse in 2D cartesian coordinate system."""
    lambdas = (
        torch.linalg.eigvalsh(conic_matrix[..., :2, :2])
        / (-torch.det(conic_matrix) / torch.det(conic_matrix[..., :2, :2]))[..., None]
    )
    axes = torch.sqrt(1 / lambdas)
    return axes[..., 0], axes[..., 1]


def ellipse_angle(conic_matrix: torch.Tensor) -> torch.Tensor:
    """Returns angle of ellipse in radians w.r.t. x-axis."""
    return (
        -torch.atan2(
            2 * conic_matrix[..., 1, 0],
            conic_matrix[..., 1, 1] - conic_matrix[..., 0, 0],
        )
        / 2
    )


def bbox_ellipse(
    ellipses: torch.Tensor,
    box_type: Literal["xyxy", "xywh", "cxcywh"] = "xyxy",
) -> torch.Tensor:
    """Converts (array of) ellipse matrices to bounding box tensor with format [xmin, ymin, xmax, ymax].

    Parameters
    ----------
    ellipses:
        Array of ellipse matrices
    box_type:
        Format of bounding boxes, default is "xyxy"

    Returns
    -------
        Array of bounding boxes
    """
    cx, cy = conic_center(ellipses)
    theta = ellipse_angle(ellipses)
    semi_major_axis, semi_minor_axis = ellipse_axes(ellipses)

    ux, uy = semi_major_axis * torch.cos(theta), semi_major_axis * torch.sin(theta)
    vx, vy = (
        semi_minor_axis * torch.cos(theta + torch.pi / 2),
        semi_minor_axis * torch.sin(theta + torch.pi / 2),
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

    if box_type != "xyxy":
        from torchvision.ops import boxes as box_ops

        bboxes = box_ops.box_convert(bboxes, in_fmt="xyxy", out_fmt=box_type)

    return bboxes
