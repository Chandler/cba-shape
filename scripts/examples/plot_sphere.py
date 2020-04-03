import sys

sys.path.append("src")

from geometry.smooth.surfaces import Sphere
from geometry.discrete import util
import open3d as o3d

surface = Sphere()

verts, faces = util.triangulate_surface(surface, 100)

triangle_mesh = o3d.geometry.TriangleMesh(
    vertices=o3d.utility.Vector3dVector(verts), triangles=o3d.utility.Vector3iVector(faces)
)

o3d.visualization.draw_geometries([triangle_mesh])
