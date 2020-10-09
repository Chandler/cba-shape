import sys

sys.path.append("src")

from color.surfaces import SpectralCone
from color.color_util import xyz_to_srgb
from graphics.blender.render import render
from geometry.discrete.util import triangulate_surface
import open3d as o3d
import numpy as np

step = 100
surface = SpectralCone(XYZ_to_outputspace=xyz_to_srgb)
verts, faces = triangulate_surface(surface, step)
colors = verts


mesh = o3d.geometry.TriangleMesh(
    vertices=o3d.utility.Vector3dVector(verts),
    triangles=o3d.utility.Vector3iVector(faces)
)
mesh.vertex_colors = o3d.utility.Vector3dVector(np.copy(colors))
mesh.compute_triangle_normals()
o3d.io.write_triangle_mesh("data/test.ply", mesh)

render(
	mesh_vertices=verts,
	mesh_faces=faces,
	mesh_colors=verts,
	light_location=(30,30,30),
	camera_location=(30,30,30),
	camera_lookat=(0,0,0),
	output_file="test.png",
	output_resolution_px=5000
)
