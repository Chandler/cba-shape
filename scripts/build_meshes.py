import sys

sys.path.append("src")

from geometry.catalog import smooth_surfaces
from geometry.discrete import util
import open3d as o3d
import numpy as np
from color.color_util import color_map

n = 100

for name, surface in smooth_surfaces.items():
    print("Building: {}".format(name))

    uv = surface.coordinates(n)

    # Choose a scalar field and color map it
    # scalar_field = np.array(surface.X(uv))
    k1, k2 = np.array(surface.principal_curvature(uv))
    scalar_field = np.abs(k1-k2)
    colors = color_map(scalar_field)

    verts, faces = util.triangulate_surface(surface, n)

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(faces))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.copy(colors))
    mesh.compute_triangle_normals()
    o3d.io.write_triangle_mesh("data/{}_mesh.ply".format(name), mesh)


