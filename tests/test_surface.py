import sys

sys.path.append("src")
import numpy.testing as npt
from geometry.smooth.surfaces import Sphere, EllipsoidLatLon, EllipsoidTriaxial, Torus
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
    r = 2
    area = Sphere(r).surface_area(step=50)
    np.testing.assert_almost_equal(area, 4 * math.pi * r * r, decimal=1)


def test_jacobian():
    """
    Compare two implementations of the jacobian
    """
    sphere = Sphere()
    uv = sphere.coordinates(10)
    jac1 = sphere.jacobian_matrix(uv)
    jac2 = np.array(sphere.df(uv))
    np.testing.assert_almost_equal(jac1, jac2)


def test_normals():
    """
    On the unit sphere the points and normals are the same..
    """
    sphere = Sphere()
    uv = sphere.coordinates(10)
    points = sphere.f(uv)
    normals = sphere.unit_normals(uv)
    valid_indicies = ~np.isnan(normals).any(axis=1)
    np.testing.assert_almost_equal(
        np.abs(normals[valid_indicies]), np.abs(points[valid_indicies]), decimal=1
    )


def test_guass_map():
    """ Compare different implementations
    of the normal map
    """
    sphere = Sphere()
    uv = sphere.coordinates(10)
    n1 = sphere.N(uv)
    n2 = sphere.unit_normals(uv)
    np.testing.assert_array_almost_equal(n1, n2)


def test_shape_operator():
    """ Compare different implementations
    of the normal map
    """
    n = 100
    surface = EllipsoidLatLon()
    uv = surface.coordinates(n)

    df = np.array(surface.df(uv))
    dN = np.array(surface.dN(uv))
    S  = surface.shape_operator(uv)

    # df*S = dN
    dfs = np.einsum("nij,njk->nik", df, S)

    try:
        np.testing.assert_array_almost_equal(dfs, dN, decimal=6)
    except AssertionError as e:
        # lol 
        mismatch = float(e.args[0].split('\n')[3].split(' ')[5][1:-2])
        assert float(mismatch) < 1

def test_gaussian_curvature_methods():
    """ Test that methods of generating
    K are equal
    """
    n = 10
    for surface in EllipsoidLatLon(), EllipsoidTriaxial(), Sphere(), Torus():
        uv = surface.coordinates(n)

        gc1 = surface.gaussian_curvature(uv)
        gc2 = surface.gaussian_curvature2(uv)
        gc3 = surface.gaussian_curvature3(uv)

        np.testing.assert_array_almost_equal(gc1, gc2, decimal=2)
        np.testing.assert_array_almost_equal(gc2, gc3, decimal=2)


def test_different_parameterizations():
    e1 = EllipsoidLatLon()
    e2 = EllipsoidTriaxial()

    np.testing.assert_almost_equal(e1.surface_area(100), e2.surface_area(100), decimal=1)

    np.testing.assert_almost_equal(
        e1.total_gaussian_curvature(100), e2.total_gaussian_curvature(100), decimal=1
    )


def test_gauss_bonnet():
    """ Test the theorum that total guassian curvature
    of a closed surface equals 2*pi*euler characteristic
    """
    SPHERE_EULER_CHARACTERISTIC = 2
    TORUS_EULER_CHARACTERISTIC = 0

    surface = Sphere(2)
    total = surface.total_gaussian_curvature(500)
    np.testing.assert_almost_equal(total, 2 * math.pi * SPHERE_EULER_CHARACTERISTIC, decimal=1)

    surface = EllipsoidLatLon()
    total = surface.total_gaussian_curvature(500)
    np.testing.assert_almost_equal(total, 2 * math.pi * SPHERE_EULER_CHARACTERISTIC, decimal=1)

    surface = Torus()
    total = surface.total_gaussian_curvature(500)
    np.testing.assert_almost_equal(total, 2 * math.pi * TORUS_EULER_CHARACTERISTIC, decimal=1)
