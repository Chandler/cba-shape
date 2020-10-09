from geometry.smooth.curves import SpaceCurve
from util import sample_range
from thirdparty.trimesh import TriMesh as TriMeshThirdParty # type: ignore
from geometry.discrete import util
from geometry.smooth.surfaces import Surface
import numpy as np
import math
import open3d

def vectorized_angle_between(vectorsA, vectorsB):
    unit_vectorsA = vectorsA / np.linalg.norm(vectorsA, axis=1)
    unit_vectorsB = vectorsB / np.linalg.norm(vectorsB, axis=1)
    return unit_vectorsB.dot(unit_vectorsA)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class TriangleMesh(TriMeshThirdParty):
    """ Build on top of the nice TriMeshThirdParty
    """
    @classmethod
    def from_surface(cls, surface: Surface, step: float):
        """ linearly rasterize a smooth SpaceCurve into a PolygonSpaceCurve
        """
        vs, faces = util.triangulate_surface(surface, step)
        return cls(vs, faces)

    @classmethod
    def read(cls, path):
        return cls.from_o3d(open3d.io.read_triangle_mesh(path))

    @classmethod
    def from_o3d(cls, o3d_mesh):
        # o3d_mesh = o3d_mesh.filter_smooth_laplacian(100)
        vs = np.array(o3d_mesh.vertices)
        faces = np.array(o3d_mesh.triangles)
        return cls(vs, faces)

    def total_gaussian_curvature(self):
        return sum(self.vertex_angle_defects())

    def euler_characteristic(self):
        """
        X = V - E + F
        https://en.wikipedia.org/wiki/Euler_characteristic
        """
        return len(self.vs) - len(self.edges) + len(self.faces)
    
    def vertex_angle_defects(self):
        """
        2pi minus the sum of the angles incident on the given vertex
        """
        defects = []
        for i in range(len(self.vs)):

            angles = []

            for k in self.vertex_face_neighbors(i):
                face  = self.faces[k]
                a     = self.vs[i]
                b,c   = [self.vs[j] for j in face if i != j]
                vec1  = np.subtract(b,a)
                vec2  = np.subtract(c,a)
                angle = angle_between(vec1, vec2)
                angles.append(angle)

            defect = (2*math.pi - sum(angles))
            defects.append(defect)
        return defects

    def vertex_curvature(self):
        """
        gather all faces around the vector
        get their normals take the average of all the normals
        """
        face_normals = self.get_face_normals()
        vector_curvatures = []
        for i, (vert, normal), in enumerate(zip(self.vs, self.get_vertex_normals())):
            face_indices = self.vertex_face_neighbors(i)
            
            neighbor_normals = [face_normals[i] for i in face_indices]
          
            vector_curvatures.append(np.mean(neighbor_normals))

        return vector_curvatures
