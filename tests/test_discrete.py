import sys

sys.path.append("src")
import numpy.testing as npt
from geometry.smooth import surfaces
import numpy as np
from geometry.discrete.surfaces import TriangleMesh
import math
from thirdparty.octasphere import octasphere, icosphere
import open3d

sphere = TriangleMesh.from_o3d(open3d.geometry.TriangleMesh.create_sphere())
torus = TriangleMesh.from_o3d(open3d.geometry.TriangleMesh.create_torus())

def test_gauss_bonnet2():
    """
    assert the relationship between gaussian
    curvature and topology
    """
    for mesh in [sphere, torus]:
        E = mesh.euler_characteristic()
        G = mesh.total_gaussian_curvature()

        Ge = E * 2 * math.pi
        
        np.testing.assert_almost_equal(G, E * 2 * math.pi, 5)

def test_gauss_bonnet1():
    """ Test the theorum that total guassian curvature
    of a closed surface equa ls 2*pi*euler characteristic
    """
    SPHERE_EULER_CHARACTERISTIC = 2
    TORUS_EULER_CHARACTERISTIC = 0

    np.testing.assert_almost_equal(
        sphere.euler_characteristic(), 
        SPHERE_EULER_CHARACTERISTIC, 
        decimal=5
    )
    np.testing.assert_almost_equal(
        torus.euler_characteristic(), 
        TORUS_EULER_CHARACTERISTIC, 
        decimal=5
    )

def test_discrete_smooth_total_gaussian_curvature():
    total_smooth   = surfaces.Sphere(2).total_gaussian_curvature(100)
    total_discrete = sphere.total_gaussian_curvature()

    np.testing.assert_almost_equal(total_smooth, total_discrete, decimal=1)


    # smooth_sphere = surfaces.Sphere()
    # discrete_sphere = TriangleMesh.from_surface(smooth_sphere, 100)

    # uv = smooth_sphere.coordinates(100)

    # smooth_curvatures = smooth_sphere.gaussian_curvature(uv)
    # discrete_cuvatures = discrete_sphere.vertex_angle_defects()
    # import pdb; pdb.set_trace()




    # surface = Sphere(2)
    # total = surface.total_gaussian_curvature(500)
    # np.testing.assert_almost_equal(total, 2 * math.pi * SPHERE_EULER_CHARACTERISTIC, decimal=1)

    # surface = EllipsoidLatLon()
    # total = surface.total_gaussian_curvature(500)
    # np.testing.assert_almost_equal(total, 2 * math.pi * SPHERE_EULER_CHARACTERISTIC, decimal=1)

    # surface = Torus()
    # total = surface.total_gaussian_curvature(500)
    # np.testing.assert_almost_equal(total, 2 * math.pi * TORUS_EULER_CHARACTERISTIC, decimal=1)
