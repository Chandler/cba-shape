import open3d
import polyscope as ps
import numpy as np

path = "/Users/chandler/Downloads/shared_training_cloud_omg_dataset_234_geometry_triangle_mesh.ply"

mesh = open3d.io.read_triangle_mesh(path)
mesh.compute_triangle_normals()

ps.init()

ps.register_surface_mesh(
	"my mesh",
	np.array(mesh.vertices),
	np.array(mesh.triangles),
	smooth_shade=True)

ps.get_surface_mesh("my mesh").add_vector_quantity(
	"normals",
	np.array(mesh.triangle_normals), 
	defined_on='faces',
	color=(0.2, 0.5, 0.5))

ps.show()