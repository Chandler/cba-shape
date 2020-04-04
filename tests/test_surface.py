import sys

sys.path.append("src")
import numpy.testing as npt
from geometry.smooth.surfaces import Sphere
from geometry.smooth.curves import Equator
import numpy as np
import math


def test_line_element_sphere():
    """
    Integrate the equator curve of a unit sphere
    and check it with the known value
    """
    sphere = Sphere()
    length = sphere.arc_length(Equator(), step=100)
    np.testing.assert_almost_equal(length, 2 * math.pi, decimal=3)


def test_area_element_sphere():
    """
    Integrate the area a unit sphere
    and check it with the known value
    """
    area = Sphere().surface_area(step=100)
    np.testing.assert_almost_equal(area, 4 * math.pi, decimal=1)


def test_jacobian():
    """
    Compare two implementations of the jacobian
    """
    sphere = Sphere()
    uv = sphere.coordinates(10)
    jac1 = sphere.jacobian_matrix(uv)
    jac2 = np.array(sphere.df(uv)).swapaxes(0, 1)
    np.testing.assert_almost_equal(jac1, jac2)


def test_guass_map():
    """ Compare different implementations
    of the normal map
    """
    sphere = Sphere()
    uv = sphere.coordinates(10)
    n1 = sphere.N(uv)
    n2 = sphere.unit_normals(uv)
    np.testing.assert_array_almost_equal(n1, n2)
