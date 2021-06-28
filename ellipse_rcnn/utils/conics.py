from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.coordinates import spherical_to_cartesian
from matplotlib.collections import EllipseCollection
from numba import njit
from numpy import linalg as LA
from scipy.spatial.distance import cdist

import src.common.constants as const
from src.common.camera import camera_matrix, projection_matrix, Camera
from src.common.coordinates import ENU_system
from src.common.robbins import load_craters, extract_robbins_dataset


def matrix_adjugate(matrix):
    """Return adjugate matrix [1].

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix

    Returns
    -------
    np.ndarray
        Adjugate of input matrix

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Adjugate_matrix
    """

    cofactor = LA.inv(matrix).T * LA.det(matrix)
    return cofactor.T


def scale_det(matrix):
    """Rescale matrix such that det(A) = 1.

    Parameters
    ----------
    matrix: np.ndarray, torch.Tensor
        Matrix input
    Returns
    -------
    np.ndarray
        Normalised matrix.
    """
    if isinstance(matrix, np.ndarray):
        return np.cbrt((1. / LA.det(matrix)))[..., None, None] * matrix
    elif isinstance(matrix, torch.Tensor):
        val = 1. / torch.det(matrix)
        return (torch.sign(val) * torch.pow(torch.abs(val), 1. / 3.))[..., None, None] * matrix


def conic_matrix(a, b, psi, x=0, y=0):
    """Returns matrix representation for crater derived from ellipse parameters

    Parameters
    ----------
    a: np.ndarray, torch.Tensor, int, float
        Semi-major ellipse axis
    b: np.ndarray, torch.Tensor, int, float
        Semi-minor ellipse axis
    psi: np.ndarray, torch.Tensor, int, float
        Ellipse angle (radians)
    x: np.ndarray, torch.Tensor, int, float
        X-position in 2D cartesian coordinate system (coplanar)
    y: np.ndarray, torch.Tensor, int, float
        Y-position in 2D cartesian coordinate system (coplanar)

    Returns
    -------
    np.ndarray, torch.Tensor
        Array of ellipse matrices
    """
    if isinstance(a, (int, float)):
        out = np.empty((3, 3))
        pkg = np
    elif isinstance(a, torch.Tensor):
        out = torch.empty((len(a), 3, 3), device=a.device, dtype=torch.float32)
        pkg = torch
    elif isinstance(a, np.ndarray):
        out = np.empty((len(a), 3, 3))
        pkg = np
    else:
        raise TypeError("Input must be of type torch.Tensor, np.ndarray, int or float.")

    A = (a ** 2) * pkg.sin(psi) ** 2 + (b ** 2) * pkg.cos(psi) ** 2
    B = 2 * ((b ** 2) - (a ** 2)) * pkg.cos(psi) * pkg.sin(psi)
    C = (a ** 2) * pkg.cos(psi) ** 2 + b ** 2 * pkg.sin(psi) ** 2
    D = -2 * A * x - B * y
    F = -B * x - 2 * C * y
    G = A * (x ** 2) + B * x * y + C * (y ** 2) - (a ** 2) * (b ** 2)

    out[:, 0, 0] = A
    out[:, 1, 1] = C
    out[:, 2, 2] = G

    out[:, 1, 0] = out[:, 0, 1] = B / 2

    out[:, 2, 0] = out[:, 0, 2] = D / 2

    out[:, 2, 1] = out[:, 1, 2] = F / 2

    return out


@njit
def conic_center_numba(A):
    a = LA.inv(A[:2, :2])
    b = np.expand_dims(-A[:2, 2], axis=-1)
    return a @ b


def conic_center(A):
    if isinstance(A, torch.Tensor):
        return (torch.inverse(A[..., :2, :2]) @ -A[..., :2, 2][..., None])[..., 0]
    elif isinstance(A, np.ndarray):
        return (LA.inv(A[..., :2, :2]) @ -A[..., :2, 2][..., None])[..., 0]
    else:
        raise TypeError("Input conics must be of type torch.Tensor or np.ndarray.")


def ellipse_axes(A):
    if isinstance(A, torch.Tensor):
        lambdas = torch.linalg.eigvalsh(A[..., :2, :2]) / (-torch.det(A) / torch.det(A[..., :2, :2]))[..., None]
        axes = torch.sqrt(1 / lambdas)
    elif isinstance(A, np.ndarray):
        lambdas = LA.eigvalsh(A[..., :2, :2]) / (-LA.det(A) / LA.det(A[..., :2, :2]))[..., None]
        axes = np.sqrt(1 / lambdas)
    else:
        raise TypeError("Input conics must be of type torch.Tensor or np.ndarray.")
    return axes[..., 1], axes[..., 0]


def ellipse_angle(A):
    if isinstance(A, torch.Tensor):
        return torch.atan2(2 * A[..., 1, 0], (A[..., 0, 0] - A[..., 1, 1])) / 2
    elif isinstance(A, np.ndarray):
        return np.arctan2(2 * A[..., 1, 0], (A[..., 0, 0] - A[..., 1, 1])) / 2
    else:
        raise TypeError("Input conics must be of type torch.Tensor or np.ndarray.")


def plot_conics(A_craters: Union[np.ndarray, torch.Tensor],
                resolution=const.CAMERA_RESOLUTION,
                figsize=(15, 15),
                plot_centers=False,
                ax=None,
                rim_color='r',
                alpha=1.):
    if isinstance(A_craters, torch.Tensor):
        A_craters = A_craters.numpy()

    a_proj, b_proj = ellipse_axes(A_craters)
    psi_proj = ellipse_angle(A_craters)
    r_pix_proj = conic_center(A_craters)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'aspect': 'equal'})

    # Set axes according to camera pixel space convention
    ax.set_xlim(0, resolution[0])
    ax.set_ylim(resolution[1], 0)

    ec = EllipseCollection(a_proj, b_proj, np.degrees(psi_proj), units='xy', offsets=r_pix_proj,
                           transOffset=ax.transData, facecolors="None", edgecolors=rim_color, alpha=alpha)
    ax.add_collection(ec)

    if plot_centers:
        crater_centers = conic_center(A_craters)
        for k, c_i in enumerate(crater_centers):
            x, y = c_i[0], c_i[1]
            if 0 <= x <= resolution[0] and 0 <= y <= resolution[1]:
                ax.text(x, y, str(k))


def generate_mask(A_craters,
                  resolution=const.CAMERA_RESOLUTION,
                  filled=False,
                  instancing=False,
                  thickness=1
                  ):
    a_proj, b_proj = map(lambda x: x / 2, ellipse_axes(A_craters))
    psi_proj = np.degrees(ellipse_angle(A_craters))
    r_pix_proj = conic_center(A_craters)

    a_proj, b_proj, psi_proj, r_pix_proj = map(lambda i: np.round(i).astype(int),
                                               (a_proj, b_proj, psi_proj, r_pix_proj))

    mask = np.zeros(resolution)

    if filled:
        thickness = -1

    for i, (a, b, x, y, psi) in enumerate(zip(a_proj, b_proj, *r_pix_proj.T, psi_proj), 1):
        mask = cv2.ellipse(mask,
                           (x, y),
                           (a, b),
                           psi,
                           0,
                           360,
                           i if instancing else 1,
                           thickness)

    return mask


def crater_camera_homography(r_craters, P_MC):
    """Calculate homography between crater-plane and camera reference frame.

    .. math:: \mathbf{H}_{C_i} =  ^\mathcal{M}\mathbf{P}_\mathcal{C_{craters}} [[H_{M_i}], [k^T]]

    Parameters
    ----------
    r_craters : np.ndarray
        (Nx)3x1 position vector of craters.
    P_MC : np.ndarray
        (Nx)3x4 projection matrix from selenographic frame to camera pixel frame.

    Returns
    -------
        (Nx)3x3 homography matrix
    """
    S = np.concatenate((np.identity(2), np.zeros((1, 2))), axis=0)
    k = np.array([0, 0, 1])[:, None]

    H_Mi = np.concatenate((np.concatenate(ENU_system(r_craters), axis=-1) @ S, r_craters), axis=-1)

    return P_MC @ np.concatenate((H_Mi, np.tile(k.T[None, ...], (len(H_Mi), 1, 1))), axis=1)


def project_crater_conics(C_craters, r_craters, fov, resolution, T_CM, r_M):
    """Project crater conics into digital pixel frame. See pages 17 - 25 from [1] for methodology.

    Parameters
    ----------
    C_craters : np.ndarray
        Nx3x3 array of crater conics
    r_craters : np.ndarray
        Nx3x1 position vector of craters.
    fov : float, Iterable
        Field-of-View angle (radians), if type is Iterable it will be interpreted as (fov_x, fov_y)
    resolution : int, Iterable
        Image resolution, if type is Iterable it will be interpreted as (res_x, res_y)
    T_CM : np.ndarray
        3x3 matrix representing camera attitude in world reference frame
    r_M : np.ndarray
        3x1 position vector of camera

    Returns
    -------
    np.ndarray
        Nx3x3 Homography matrix H_Ci

    References
    ----------
    .. [1] Christian, J. A., Derksen, H., & Watkins, R. (2020). Lunar Crater Identification in Digital Images. https://arxiv.org/abs/2009.01228
    """

    K = camera_matrix(fov, resolution)
    P_MC = projection_matrix(K, T_CM, r_M)
    H_Ci = crater_camera_homography(r_craters, P_MC)
    return LA.inv(H_Ci).transpose((0, 2, 1)) @ C_craters @ LA.inv(H_Ci)


def project_crater_centers(r_craters, fov, resolution, T_CM, r_M):
    """Project crater centers into digital pixel frame.

    Parameters
    ----------
    r_craters : np.ndarray
        Nx3x1 position vector of craters.
    fov : int, float, Iterable
        Field-of-View angle (radians), if type is Iterable it will be interpreted as (fov_x, fov_y)
    resolution : int, Iterable
        Image resolution, if type is Iterable it will be interpreted as (res_x, res_y)
    T_CM : np.ndarray
        3x3 matrix representing camera attitude in world reference frame
    r_M : np.ndarray
        3x1 position vector of camera

    Returns
    -------
    np.ndarray
        Nx2x1 2D positions of craters in pixel frame
    """

    K = camera_matrix(fov, resolution)
    P_MC = projection_matrix(K, T_CM, r_M)
    H_Ci = crater_camera_homography(r_craters, P_MC)
    return (H_Ci @ np.array([0, 0, 1]) / (H_Ci @ np.array([0, 0, 1]))[:, -1][:, None])[:, :2]


class ConicProjector(Camera):
    def project_crater_conics(self, C_craters, r_craters):
        H_Ci = crater_camera_homography(r_craters, self.projection_matrix)
        return LA.inv(H_Ci).transpose((0, 2, 1)) @ C_craters @ LA.inv(H_Ci)

    def project_crater_centers(self, r_craters):
        H_Ci = crater_camera_homography(r_craters, self.projection_matrix)
        return (H_Ci @ np.array([0, 0, 1]) / (H_Ci @ np.array([0, 0, 1]))[:, -1][:, None])[:, :2]

    def generate_mask(self,
                      A_craters=None,
                      C_craters=None,
                      r_craters=None,
                      **kwargs
                      ):

        if A_craters is None:
            if C_craters is None or r_craters is None:
                raise ValueError("Must provide either crater data in respective ENU-frame (C_craters & r_craters) "
                                 "or in image-frame (A_craters)!")

            A_craters = self.project_crater_conics(C_craters, r_craters)

        return generate_mask(A_craters=A_craters, resolution=self.resolution, **kwargs)

    def plot(self,
             A_craters=None,
             C_craters=None,
             r_craters=None,
             **kwargs
             ):
        if A_craters is None:
            if C_craters is None or r_craters is None:
                raise ValueError("Must provide either crater data in respective ENU-frame (C_craters & r_craters) "
                                 "or in image-frame (A_craters)!")

            A_craters = self.project_crater_conics(C_craters, r_craters)

        plot_conics(A_craters=A_craters, resolution=self.resolution, **kwargs)


class MaskGenerator(ConicProjector):
    def __init__(self,
                 r_craters_catalogue: np.ndarray,
                 C_craters_catalogue: np.ndarray,
                 axis_threshold=const.AXIS_THRESHOLD,
                 filled=False,
                 instancing=True,
                 mask_thickness=1,
                 mask_margin=0,
                 **kwargs
                 ):
        super(MaskGenerator, self).__init__(**kwargs)

        self.mask_margin = mask_margin
        self.axis_threshold = axis_threshold
        self.mask_thickness = mask_thickness
        self.instancing = instancing
        self.filled = filled
        self.C_craters_catalogue = C_craters_catalogue
        self.r_craters_catalogue = r_craters_catalogue

    @classmethod
    def from_robbins_dataset(cls,
                             file_path="data/lunar_crater_database_robbins_2018.csv",
                             diamlims=const.DIAMLIMS,
                             ellipse_limit=const.MAX_ELLIPTICITY,
                             arc_lims=const.ARC_LIMS,
                             axis_threshold=const.AXIS_THRESHOLD,
                             filled=False,
                             instancing=True,
                             mask_thickness=1,
                             position=None,
                             resolution=const.CAMERA_RESOLUTION,
                             fov=const.CAMERA_FOV,
                             primary_body_radius=const.RMOON,
                             **load_crater_kwargs
                             ):
        lat_cat, long_cat, major_cat, minor_cat, psi_cat, crater_id = extract_robbins_dataset(
            load_craters(file_path, diamlims=diamlims, ellipse_limit=ellipse_limit, arc_lims=arc_lims,
                         **load_crater_kwargs)
        )
        r_craters_catalogue = np.array(np.array(spherical_to_cartesian(const.RMOON, lat_cat, long_cat))).T[..., None]
        C_craters_catalogue = conic_matrix(major_cat, minor_cat, psi_cat)

        return cls(r_craters_catalogue=r_craters_catalogue,
                   C_craters_catalogue=C_craters_catalogue,
                   axis_threshold=axis_threshold,
                   filled=filled,
                   instancing=instancing,
                   mask_thickness=mask_thickness,
                   resolution=resolution,
                   fov=fov,
                   primary_body_radius=primary_body_radius,
                   position=position
                   )

    def _visible(self):
        return (cdist(self.r_craters_catalogue.squeeze(), self.position.T) <=
                np.sqrt(2 * self.height * self._primary_body_radius + self.height ** 2)).ravel()

    def visible_catalogue_craters(self, margin=None):
        r_craters = self.r_craters_catalogue[self._visible()]
        C_craters = self.C_craters_catalogue[self._visible()]

        r_craters_img = self.project_crater_centers(r_craters)

        if margin is None:
            margin = self.mask_margin

        in_image = np.logical_and.reduce(
            np.logical_and(r_craters_img > -margin, r_craters_img < self.resolution[0] + margin),
            axis=1)

        r_craters = r_craters[in_image]
        C_craters = C_craters[in_image]

        return C_craters, r_craters

    def craters_in_image(self, margin=None):
        C_craters, r_craters = self.visible_catalogue_craters(margin=margin)

        A_craters = self.project_crater_conics(C_craters, r_craters)

        a_proj, b_proj = ellipse_axes(A_craters)
        axis_filter = np.logical_and(a_proj >= self.axis_threshold[0], b_proj >= self.axis_threshold[0])
        axis_filter = np.logical_and(axis_filter,
                                     np.logical_and(a_proj <= self.axis_threshold[1], b_proj <= self.axis_threshold[1]))

        return A_craters[axis_filter]

    def generate_mask(self, **kwargs):
        mask_args = dict(
            filled=self.filled,
            instancing=self.instancing,
            thickness=self.mask_thickness
        )
        mask_args.update(kwargs)

        return super(MaskGenerator, self).generate_mask(A_craters=self.craters_in_image(),
                                                        **mask_args)

    def plot(self, *args, **kwargs):
        super(MaskGenerator, self).plot(A_craters=self.craters_in_image(), *args, **kwargs)
