# import sys
# import open3d
# import numpy as np
# sys.path.append("/Users/cbabraham/code/gc-polyscope-project-template/build/lib")
# import gc_bindings

import polyscope
import igl
import numpy as np
import polyscope as ps
import scipy

# https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/spot.zip
path = "/Users/cbabraham/data/mesh/spot/spot_triangulated.obj"

# verts and faces
V,F = igl.read_triangle_mesh(path)

# cotan laplacian and mass matrix
L = igl.cotmatrix(V,F)
M = igl.massmatrix(V,F)

# eigenvectors of the laplacian
_, E = scipy.sparse.linalg.eigsh(L, 1000, M, which='BE')

C = V.T.dot(E) # inner product of each eigenvector and the mesh
R = np.einsum('km,nm->nk',C,E)

# V: n x k verts (k = 3)
# E: n x m eigenvectors, to use as bases
# C: k x m basis weights
# R: n x k synthesized vertices from weighted bases

ps.init()
original = ps.register_surface_mesh("mesh", V,F)

for i, eigenvector in enumerate(E.T):
	original.add_scalar_quantity(
		"eigenvector-{}".format(i),
		eigenvector.astype(np.float32), # 
		defined_on='vertices'
	)
test = ps.register_surface_mesh("m2", R.astype(np.float32), F)
ps.show()