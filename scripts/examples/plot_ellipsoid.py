import sys

sys.path.append("src")

from geometry.smooth.surfaces import Ellipsoid, EllipsoidSpecial, Torus
from geometry.discrete import util
import open3d as o3d
from graphics.color_util import color_map
import numpy
import numpy as np

n = 100

surface = EllipsoidSpecial()
verts, faces = util.triangulate_surface(surface, n)
triangle_mesh = o3d.geometry.TriangleMesh(
    vertices=o3d.utility.Vector3dVector(verts), triangles=o3d.utility.Vector3iVector(faces)
)

uv = surface.coordinates(n)
PC = surface.principal_curvature(uv)
pc_alignments = np.abs(PC[:, 0] - PC[:, 1])
colors = color_map(pc_alignments)

import pdb

pdb.set_trace()
triangle_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([triangle_mesh])
