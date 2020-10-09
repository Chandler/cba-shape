from geometry.smooth import surfaces
from geometry.smooth import curves
from geometry.discrete.surfaces import TriangleMesh
from thirdparty.octasphere import octasphere, icosphere

smooth_surfaces = {
    "unit_sphere": surfaces.Sphere(0.3),
    "elliptical_torus": surfaces.EllipticalTorus(),
    "torus": surfaces.Torus(),
    "ellipsoid": surfaces.EllipsoidLatLon(),
    "cap": surfaces.Hemisphere(1),
    "cup": surfaces.Hemisphere(-1),
    "ellipsoid_triaxial": surfaces.EllipsoidTriaxial(),
    "helix_tube": surfaces.SpaceCurveTube(curve=curves.Helix([0.0, 10.0]), radius=0.1),
    "treifoil_tube": surfaces.SpaceCurveTube(curve=curves.TrefoilKnot(), radius=6),
    "dini_surface": surfaces.GeneralizedHelicoid(curve=curves.Tractrix(1), slant=0.15),
    "monkey_saddle": surfaces.MonkeySaddle(),
    "catenoid": surfaces.Catenoid(),
    "cylinder": surfaces.SurfaceOfRevolution(curves.Line2D()),
    "plane": surfaces.Plane()
}

smooth_curves = {
	"helix": curves.Helix([0.0, 10.0])
}

# discrete_surfaces = {
#     "octasphere": TriangleMesh(**octasphere(5, 1)),
#     "icosphere": TriangleMesh(**icosphere())
# }