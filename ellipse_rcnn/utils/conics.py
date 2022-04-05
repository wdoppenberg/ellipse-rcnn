from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch

pi = 3.141592653589793


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


def unimodular_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Rescale matrix such that det(ellipses) = 1, in other words, make it unimodular.

    Parameters
    ----------
    matrix:
        Matrix input

    Returns
    -------
    torch.Tensor
        Unimodular version of input matrix.
    """
    val = 1. / torch.det(matrix)
    return (torch.sign(val) * torch.pow(torch.abs(val), 1. / 3.))[..., None, None] * matrix


def conic_matrix(
        a: Union[torch.Tensor, float],
        b: Union[torch.Tensor, float],
        psi: Union[torch.Tensor, float],
        x: Union[torch.Tensor, float] = 0,
        y: Union[torch.Tensor, float] = 0
) -> torch.Tensor:
    """Returns matrix representation for crater derived from ellipse parameters

    Parameters
    ----------
    a:
        Semi-major ellipse axis
    b:
        Semi-minor ellipse axis
    psi:
        Ellipse angle (radians)
    x:
        X-position in 2D cartesian coordinate system (coplanar)
    y:
        Y-position in 2D cartesian coordinate system (coplanar)

    Returns
    -------
    torch.Tensor
        Array of ellipse matrices
    """
    if isinstance(a, (int, float)):
        a, b, psi, x, y = map(torch.Tensor, ([a], [b], [psi], [x], [y]))
    out = torch.empty((len(a), 3, 3), device=a.device, dtype=torch.float32)

    A = (a ** 2) * torch.sin(psi) ** 2 + (b ** 2) * torch.cos(psi) ** 2
    B = 2 * ((b ** 2) - (a ** 2)) * torch.cos(psi) * torch.sin(psi)
    C = (a ** 2) * torch.cos(psi) ** 2 + b ** 2 * torch.sin(psi) ** 2
    D = -2 * A * x - B * y
    F = -B * x - 2 * C * y
    G = A * (x ** 2) + B * x * y + C * (y ** 2) - (a ** 2) * (b ** 2)

    out[..., 0, 0] = A
    out[..., 1, 1] = C
    out[..., 2, 2] = G

    out[..., 1, 0] = out[..., 0, 1] = B / 2

    out[..., 2, 0] = out[..., 0, 2] = D / 2

    out[..., 2, 1] = out[..., 1, 2] = F / 2

    return out.squeeze()


def conic_center(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns center of ellipse in 2D cartesian coordinate system."""
    centers = (torch.inverse(A[..., :2, :2]) @ -A[..., :2, 2][..., None]).squeeze()
    return centers[..., 0], centers[..., 1]


def ellipse_axes(A: torch.Tensor):
    """Returns semi-major and semi-minor axes of ellipse in 2D cartesian coordinate system."""
    lambdas = torch.linalg.eigvalsh(A[..., :2, :2]) / (-torch.det(A) / torch.det(A[..., :2, :2]))[..., None]
    axes = torch.sqrt(1 / lambdas)
    return axes[..., 0], axes[..., 1]


def ellipse_angle(A: torch.Tensor):
    """Returns angle of ellipse in radians w.r.t. x-axis."""
    return torch.atan2(2 * A[..., 1, 0], (A[..., 0, 0] - A[..., 1, 1])) / 2


def bbox_ellipse(ellipses: torch.Tensor) -> torch.Tensor:
    """Converts (array of) ellipse matrices to bounding box tensor with format [xmin, ymin, xmax, ymax].

    :param ellipses:
        Array of ellipse matrices
    :return:
        Array of bounding boxes
    """
    cx, cy = conic_center(ellipses)
    psi = ellipse_angle(ellipses)
    a, b = ellipse_axes(ellipses)

    ux, uy = a * torch.cos(psi), a * torch.sin(psi)
    vx, vy = b * torch.cos(psi + pi / 2), b * torch.sin(psi + pi / 2)

    box_width = torch.sqrt(ux ** 2 + vx ** 2)
    box_height = torch.sqrt(uy ** 2 + vy ** 2)

    boxes_analytical = torch.vstack(
        (cx - box_width / 2, cy - box_height / 2, cx + box_width / 2, cy + box_height / 2)).T.to(ellipses)

    return boxes_analytical


class EllipseBase(ABC):
    @abstractmethod
    def __init__(self, conic_matrix: torch.Tensor) -> None:
        self._data = conic_matrix

    @classmethod
    def from_params(cls, a, b, psi, x=0, y=0) -> EllipseBase:
        """Creates ellipse from parametric representation."""
        return cls(conic_matrix(a, b, psi, x, y))

    @property
    def axes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns semi-major and semi-minor axes of ellipse in 2D cartesian coordinate system."""
        return ellipse_axes(self.matrix)

    @property
    def angle(self) -> torch.Tensor:
        """Returns angle of ellipse in radians w.r.t. x-axis."""
        return ellipse_angle(self.matrix)

    @property
    def center(self):
        """Returns center of ellipse in 2D cartesian coordinate system."""
        return conic_center(self.matrix)

    @property
    def matrix(self):
        """Returns ellipse matrix."""
        return self._data

    @matrix.setter
    def matrix(self, other: torch.Tensor):
        """Sets ellipse matrix."""
        if other.shape != torch.Size([3, 3]):
            raise ValueError("Input array needs to be 3x3!")
        self._data = other

    def to(self, *args, **kwargs):
        """Move underlying data to specified device."""
        self._data.to(*args, **kwargs)

    def device(self):
        """Returns device of underlying data."""
        return self._data.device

    def __str__(self):
        """Returns string representation of ellipse."""
        a, b = self.axes
        x, y = self.center
        angle = self.angle
        return f"Ellipse<a={a:.1f}, b={b:.1f}, angle={angle:.1f}, x={x:.1f}, y={y:.1f}, device={self.device}>"

    def __repr__(self):
        """Returns string representation of ellipse."""
        return str(self)

    def __del__(self):
        """Deletes underlying data."""
        del self._data


class Ellipse(EllipseBase):
    """Class for ellipse representation."""

    def __init__(self, conic_matrix: torch.Tensor):
        if conic_matrix.shape != torch.Size([3, 3]):
            raise ValueError("Input array needs to be 3x3!")
        self._data: torch.Tensor = conic_matrix


class EllipseCollection(EllipseBase):
    def __init__(self, conic_matrices: torch.Tensor):
        if conic_matrices.shape[-2:] != torch.Size([3, 3]) or len(conic_matrices.shape) > 3:
            raise ValueError("Input array needs to be Nx3x3!")
        self._data: torch.Tensor = conic_matrices
