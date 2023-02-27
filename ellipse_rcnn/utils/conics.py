from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch

pi = 3.141592653589793
ZERO_TENSOR = torch.tensor(0.0)


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
    return (torch.sign(val) * torch.pow(torch.abs(val), 1.0 / 3.0))[..., None, None] * matrix


@torch.jit.script
def ellipse_to_conic_matrix(
    a: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor = ZERO_TENSOR,
    y: torch.Tensor = ZERO_TENSOR,
    theta: torch.Tensor = ZERO_TENSOR,
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

    # conic_matrix = torch.empty((len(a), 3, 3), device=a.device, dtype=a.dtype)
    # conic_matrix.requires_grad_(a.requires_grad)

    A = (a**2) * torch.sin(theta) ** 2 + (b**2) * torch.cos(theta) ** 2
    B = 2 * ((b**2) - (a**2)) * torch.cos(theta) * torch.sin(theta)
    C = (a**2) * torch.cos(theta) ** 2 + b**2 * torch.sin(theta) ** 2
    D = -2 * A * x - B * y
    F = -B * x - 2 * C * y
    G = A * (x**2) + B * x * y + C * (y**2) - (a**2) * (b**2)

    # Create (array of) of conic matrix (N, 3, 3)
    conic_matrix = torch.stack(
        tensors=(
            torch.stack((A, B / 2, D / 2), dim=-1),
            torch.stack((B / 2, C, F / 2), dim=-1),
            torch.stack((D / 2, F / 2, G), dim=-1)
        ),
        dim=-1
        )

    return conic_matrix.squeeze()


def conic_center(conic_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns center of ellipse in 2D cartesian coordinate system."""
    centers = (torch.inverse(conic_matrix[..., :2, :2]) @ -conic_matrix[..., :2, 2][..., None]).squeeze()
    return centers[..., 0], centers[..., 1]


def ellipse_axes(conic_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns major and minor axes of ellipse in 2D cartesian coordinate system."""
    lambdas = (
        torch.linalg.eigvalsh(conic_matrix[..., :2, :2])
        / (-torch.det(conic_matrix) / torch.det(conic_matrix[..., :2, :2]))[..., None]
    )
    axes = torch.sqrt(1 / lambdas)
    return axes[..., 0], axes[..., 1]


def ellipse_semi_axes(conic_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns semi-major and semi-minor axes of ellipse in 2D cartesian coordinate system."""
    major_axis, minor_axis = ellipse_axes(conic_matrix)
    return major_axis / 2, minor_axis / 2


def ellipse_angle(conic_matrix: torch.Tensor) -> torch.Tensor:
    """Returns angle of ellipse in radians w.r.t. x-axis."""
    return -torch.atan2(2 * conic_matrix[..., 1, 0], conic_matrix[..., 1, 1] - conic_matrix[..., 0, 0]) / 2


def bbox_ellipse(ellipses: torch.Tensor) -> torch.Tensor:
    """Converts (array of) ellipse matrices to bounding box tensor with format [xmin, ymin, xmax, ymax].

    Parameters
    ----------
    ellipses:
        Array of ellipse matrices

    Returns
    -------
        Array of bounding boxes
    """
    cx, cy = conic_center(ellipses)
    theta = ellipse_angle(ellipses)
    semi_major_axis, semi_minor_axis = ellipse_axes(ellipses)

    ux, uy = semi_major_axis * torch.cos(theta), semi_major_axis * torch.sin(theta)
    vx, vy = semi_minor_axis * torch.cos(theta + pi / 2), semi_minor_axis * torch.sin(theta + pi / 2)

    box_halfwidth = torch.sqrt(ux**2 + vx**2)
    box_halfheight = torch.sqrt(uy**2 + vy**2)

    bboxes = torch.vstack((cx - box_halfwidth, cy - box_halfheight, cx + box_halfwidth, cy + box_halfheight)).T.to(
        ellipses
    )

    return bboxes


class EllipseBase(ABC):
    @abstractmethod
    def __init__(self, conic_matrix: torch.Tensor) -> None:
        self._data = conic_matrix

    @classmethod
    def from_params(
        cls,
        a: torch.Tensor,
        b: torch.Tensor,
        psi: torch.Tensor,
        x: torch.Tensor = ZERO_TENSOR,
        y: torch.Tensor = ZERO_TENSOR,
    ) -> EllipseBase:
        """Creates ellipse from parametric representation."""
        return cls(ellipse_to_conic_matrix(a, b, x, y, psi))

    @property
    def axes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns semi-major and semi-minor axes of ellipse in 2D cartesian coordinate system."""
        return ellipse_axes(self.matrix)

    @property
    def angle(self) -> torch.Tensor:
        """Returns angle of ellipse in radians w.r.t. x-axis."""
        return ellipse_angle(self.matrix)

    @property
    def center(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns center of ellipse in 2D cartesian coordinate system."""
        return conic_center(self.matrix)

    @property
    def matrix(self) -> torch.Tensor:
        """Returns ellipse matrix."""
        return self._data

    @matrix.setter
    def matrix(self, other: torch.Tensor) -> None:
        """Sets ellipse matrix."""
        if other.shape != torch.Size([3, 3]):
            raise ValueError("Input array needs to be 3x3!")
        self._data = other

    def to(self, *args: Any, **kwargs: Any) -> None:
        """Move underlying data to specified device."""
        self._data.to(*args, **kwargs)

    def device(self) -> torch.device:
        """Returns device of underlying data."""
        return self._data.device

    def __str__(self) -> str:
        """Returns string representation of ellipse."""
        a, b = self.axes
        x, y = self.center
        angle = self.angle
        return f"Ellipse<a={a:.1f}, b={b:.1f}, angle={angle:.1f}, x={x:.1f}, y={y:.1f}, device={self.device}>"

    def __repr__(self) -> str:
        """Returns string representation of ellipse."""
        return str(self)

    def __del__(self) -> None:
        """Deletes underlying data."""
        del self._data


class Ellipse(EllipseBase):
    """Class for ellipse representation."""

    def __init__(self, conic_matrix: torch.Tensor):
        if conic_matrix.shape != torch.Size([3, 3]):
            raise ValueError("Input array needs to be 3x3!")
        self._data: torch.Tensor = conic_matrix


class EllipseCollection(EllipseBase):
    """Class for ellipse collection representation."""

    def __init__(self, conic_matrices: torch.Tensor):
        if conic_matrices.shape[-2:] != torch.Size([3, 3]) or len(conic_matrices.shape) > 3:
            raise ValueError("Input array needs to be Nx3x3!")
        self._data: torch.Tensor = conic_matrices
