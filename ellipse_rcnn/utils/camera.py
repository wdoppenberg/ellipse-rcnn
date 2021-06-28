from collections.abc import Iterable
from typing import Tuple

import numpy as np
import numpy.linalg as LA
from astropy.coordinates import spherical_to_cartesian

import src.common.constants as const
from src.common.coordinates import OrbitingBodyBase


def camera_matrix(fov=const.CAMERA_FOV, resolution=const.CAMERA_RESOLUTION, alpha=0):
    """Returns camera matrix [1] from Field-of-View, skew, and offset.

    Parameters
    ----------
    fov : float, Iterable
        Field-of-View angle (degrees), if type is Iterable it will be interpreted as (fov_x, fov_y)
    resolution : float, Iterable
        X- and Y-resolution of the image in pixels
    alpha : float
        Camera skew angle.

    Returns
    -------
    np.ndarray
        3x3 camera matrix

    References
    ----------
    .. [1] https://www.cs.ucf.edu/~mtappen/cap5415/lecs/lec19.pdf
    """

    if isinstance(resolution, Iterable):
        x_0, y_0 = map(lambda num: num / 2, resolution)
    else:
        x_0 = y_0 = resolution / 2

    if isinstance(fov, Iterable):
        f_x, f_y = map(lambda num, fov_: num / np.tan(np.radians(fov_ / 2)), (x_0, y_0), fov)
    else:
        f_x, f_y = map(lambda num: num / np.tan(np.radians(fov / 2)), (x_0, y_0))

    return np.array([[f_x, alpha, x_0],
                     [0, f_y, y_0],
                     [0, 0, 1]])


def projection_matrix(K, T_CM, r_M):
    """Return Projection matrix [1] according to:

    .. math:: \mathbf{P}_C = \mathbf{K} [ \mathbf{T^C_M} & -r_C]

    Parameters
    ----------
    K : np.ndarray
        3x3 camera matrix
    T_CM : np.ndarray
        3x3 attitude matrix of camera in selenographic frame.
    r_M : np.ndarray
        3x1 camera position in world reference frame

    Returns
    -------
    np.ndarray
        3x4 projection matrix

    References
    ----------
    .. [1] https://www.cs.ucf.edu/~mtappen/cap5415/lecs/lec19.pdf

    See Also
    --------
    camera_matrix

    """
    return K @ LA.inv(T_CM) @ np.concatenate((np.identity(3), -r_M), axis=1)


class Camera(OrbitingBodyBase):
    """
    Camera data class with associated state attributes & functions.
    """

    def __init__(self, fov=const.CAMERA_FOV, resolution=const.CAMERA_RESOLUTION, **kwargs):
        super().__init__(**kwargs)
        self.__fov = None
        self.__resolution = None

        self.fov = fov
        self.resolution = resolution

    @classmethod
    def from_coordinates(cls,
                         lat,
                         long,
                         height,
                         fov=const.CAMERA_FOV,
                         resolution=const.CAMERA_RESOLUTION,
                         attitude=None,
                         Rbody=const.RMOON,
                         convert_to_radians=False
                         ):
        if convert_to_radians:
            lat, long = map(np.radians, (lat, long))

        position = np.array(spherical_to_cartesian(Rbody + height, lat, long))
        return cls(position=position, attitude=attitude, fov=fov, resolution=resolution, primary_body_radius=Rbody)

    @property
    def fov(self) -> Tuple:
        return self.__fov

    @fov.setter
    def fov(self, fov):
        """
        Set instance's Field-of-View in radians.

        Parameters
        ----------
        fov: int, float, Iterable
            Field-of-View angle (radians), if type is Iterable it will be interpreted as (fov_x, fov_y)
        """
        if not isinstance(fov, Iterable):
            self.__fov = (fov, fov)
        else:
            self.__fov = tuple(fov)

    @property
    def resolution(self) -> Tuple:
        return self.__resolution

    @resolution.setter
    def resolution(self, resolution):
        """
        Set instance's resolution in pixels.

        Parameters
        ----------
        resolution : int, Iterable
            Image resolution, if type is Iterable it will be interpreted as (res_x, res_y)
        """
        if not isinstance(resolution, Iterable):
            self.__resolution = (resolution, resolution)
        else:
            self.__resolution = tuple(resolution)

    @property
    def camera_matrix(self) -> np.ndarray:
        return camera_matrix(fov=self.fov, resolution=self.resolution)

    # Alias
    K: np.ndarray = camera_matrix

    @property
    def projection_matrix(self) -> np.ndarray:
        return projection_matrix(K=self.K, T_CM=self.attitude, r_M=self.position)

    # Alias
    P: np.ndarray = projection_matrix
