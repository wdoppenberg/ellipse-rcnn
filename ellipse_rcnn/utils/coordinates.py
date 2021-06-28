from typing import Union

import numpy as np
import numpy.linalg as LA
from astropy.coordinates import spherical_to_cartesian, cartesian_to_spherical
from scipy.spatial.transform import Rotation

import src.common.constants as const


def ENU_system(r):
    """Return local East-North-Up (ENU) coordinate system for point defined by p.

    Using the coordinate system defined using:

    .. math::
        \mathbf{u}_i = \mathbf{p}^{(c)}_{M_i}/||\mathbf{p}^{(c)}_{M_i}||

        \mathbf{e}_i = cross(\mathbf{k}, \mathbf{u}_i )/|| cross(\mathbf{k}, \mathbf{u}_i) ||

        \mathbf{n}_i = cross(\mathbf{u}_i, \mathbf{e}_i)/|| cross(\mathbf{u}_i, \mathbf{e}_i) ||

    with

    .. math::
        \mathbf{k} = [0 & 0 & 1]^T

    and :math:`p_{Mi}` is the selenographic 3D cartesian coordinate derived from latitude & longitude.

    Parameters
    ----------
    r : np.ndarray
        (Nx)3x1 vector that defines origin.

    Returns
    -------
    e_i, n_i, u_i : np.ndarray
        Normalized i, j, k components of coordinate system.

    """
    k = np.array([0, 0, 1])[:, None]

    u_i = r / LA.norm(r, ord=2, axis=(-1, -2), keepdims=True)

    e_i = np.cross(k, r, axisa=0, axisb=-2, axisc=-2)
    e_i /= LA.norm(e_i, ord=2, axis=(-1, -2), keepdims=True)

    n_i = np.cross(r, e_i, axis=-2)
    n_i /= LA.norm(n_i, ord=2, axis=(-1, -2), keepdims=True)

    return e_i, n_i, u_i


def nadir_attitude(r):
    """Return nadir-pointing (z-axis) coordinate system for point defined by r in world reference frame. X- and
    Y-components are defined by East and South respectively.

    Parameters
    ----------
    r : np.ndarray
        (Nx)3x1 vector that defines origin.

    Returns
    -------
    e_i, n_i, d_i : np.ndarray
        Normalized i, j, k components of coordinate system.

    """
    k = np.array([0, 0, 1])[:, None]
    d_i = -r / LA.norm(r, ord=2, axis=(-1, -2), keepdims=True)

    e_i = np.cross(k, -d_i, axisa=0, axisb=-2, axisc=-2)
    e_i /= LA.norm(e_i, ord=2, axis=(-1, -2), keepdims=True)

    s_i = np.cross(d_i, e_i, axis=-2)
    s_i /= LA.norm(s_i, ord=2, axis=(-1, -2), keepdims=True)

    return e_i, s_i, d_i


def suborbital_coords(r, R_body=const.RMOON):
    """Return coordinates directly below orbital position.

    Parameters
    ----------
    r : np.ndarray
        Position above body (e.g. Moon)
    R_body : np.ndarray
        Radius of body in km, defaults to const.RMOON

    Returns
    -------
    np.ndarray
        Suborbital coordinates
    """
    return (r / LA.norm(r)) * R_body


class OrbitingBodyBase:
    """
    Base class implementing all positional and orientation attributes + methods.
    """
    def __init__(self,
                 position=None,
                 attitude=None,
                 primary_body_radius=const.RMOON
                 ):

        self._primary_body_radius = primary_body_radius
        self.__position = None
        self.__attitude = None

        self.position = position
        self.attitude = attitude

    @classmethod
    def from_coordinates(cls,
                         lat,
                         long,
                         height,
                         attitude=None,
                         Rbody=const.RMOON,
                         convert_to_radians=False
                         ):
        if convert_to_radians:
            lat, long = map(np.radians, (lat, long))

        position = np.array(spherical_to_cartesian(Rbody + height, lat, long))
        return cls(position=position, attitude=attitude, primary_body_radius=Rbody)

    @property
    def position(self) -> np.ndarray:
        return self.__position

    @position.setter
    def position(self, position: np.ndarray):
        """
        Sets instance's position in Cartesian space.

        If set to None, a random position above the moon will be generated between 150 and 400 km height.

        Parameters
        ----------
        position : np.ndarray
            3x1 position vector of camera.
        """
        if position is None:
            self.set_coordinates(0, 0, 250)
        else:
            # Ensure 3x1 vector
            if len(position.shape) == 1:
                position = position[:, None]
            elif len(position.shape) > 2:
                raise ValueError("Position vector must be 1 or 2-dimensional (3x1)!")

            if LA.norm(position) < self._primary_body_radius:
                raise ValueError(
                    f"New position vector is inside the Moon! (Distance to center = {LA.norm(position):.2f} km, "
                    f"R_moon = {self._primary_body_radius})"
                )

            if not position.dtype == np.float64:
                position = position.astype(np.float64)

            self.__position = position

    def set_coordinates(self,
                        lat,
                        long,
                        height=None,
                        point_nadir=False,
                        convert_to_radians=True
                        ):
        if height is None:
            height = self.height

        if convert_to_radians:
            lat, long = map(np.radians, (lat, long))

        self.position = np.array(spherical_to_cartesian(self._primary_body_radius + height, lat, long))

        if point_nadir:
            self.point_nadir()

    def set_random_position(self, min_height=150, max_height=400, height=None):
        lat = np.random.randint(-90, 90)
        long = np.random.randint(-180, 180)
        if height is None:
            height = np.random.randint(min_height, max_height)
        self.set_coordinates(lat, long, height, point_nadir=True, convert_to_radians=True)

    @property
    def attitude(self) -> np.ndarray:
        return self.__attitude

    @attitude.setter
    def attitude(self, attitude: Union[np.ndarray, Rotation]):
        """
        Sets instance's attitude

        Parameters
        ----------
        attitude : np.ndarray, Rotation
            Orientation / attitude matrix (3x3) or scipy.spatial.transform.Rotation
        """
        if attitude is None:
            self.point_nadir()
        else:
            if isinstance(attitude, Rotation):
                attitude = attitude.as_matrix()

            if not np.isclose(abs(LA.det(attitude)), 1):
                raise ValueError(f"Invalid rotation matrix! Determinant should be +-1, is {LA.det(attitude)}.")

            if LA.matrix_rank(attitude) != 3:
                raise ValueError("Invalid camera attitude matrix!:\n", attitude)

            self.__attitude = attitude

    @property
    def quaternion(self):
        return Rotation.from_matrix(self.attitude).as_quat()

    @quaternion.setter
    def quaternion(self, quaternion):
        self.attitude = Rotation.from_quat(quaternion)

    # Aliases
    r: np.ndarray = position
    T: np.ndarray = attitude
    q: np.ndarray = quaternion

    @property
    def coordinates(self):
        return tuple(map(lambda x: x.value.item(), cartesian_to_spherical(*self.position)))

    @property
    def latitude(self):
        _, lat, _ = self.coordinates
        return np.degrees(lat)

    @property
    def longitude(self):
        _, _, long = self.coordinates
        return np.degrees(long)

    @property
    def height(self):
        return LA.norm(self.position) - self._primary_body_radius

    @height.setter
    def height(self, height):
        """
        Adjusts radial height without changing angular position.

        Parameters
        ----------
        height: int, float
            Height to set to in km.
        """
        if height <= 0:
            raise ValueError(f"Height cannot be below 0! (height = {height})")

        self.position = (self.position / LA.norm(self.position)) * (self._primary_body_radius + height)

    def rotate(self, axis: str, angle: float, degrees: bool = True, reset_first: bool = False):
        if axis not in ('x', 'y', 'z', 'pitch', 'yaw', 'roll'):
            raise ValueError("axis must be 'x', 'y', 'z', or 'pitch', 'yaw', 'roll'")

        if axis == 'roll':
            axis = 'z'
        elif axis == 'pitch':
            axis = 'x'
        elif axis == 'yaw':
            axis = 'y'

        if reset_first:
            self.point_nadir()

        self.attitude = (
                Rotation.from_matrix(self.attitude) * Rotation.from_euler(axis, angle, degrees=degrees)
        ).as_matrix()

    def point_nadir(self):
        self.attitude = np.concatenate(nadir_attitude(self.position), axis=-1)

    def suborbital_position(self):
        return suborbital_coords(self.r, self._primary_body_radius)
