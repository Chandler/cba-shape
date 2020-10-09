
import sys

sys.path.append("src")

from geometry.catalog import smooth_surfaces
from geometry.discrete import util
import polyscope as ps
import argparse
import numpy as np
from color.color_util import web_colors
import colour
from color.shape_scale import shape_scale_to_color
import jax

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--names', nargs='+', help='<Required> Set flag', required=True)

    args = parser.parse_args()

    ps.init()

    for name in args.names:
        n = 100
        surface = smooth_surfaces[name]
        verts, faces = util.triangulate_surface(surface, n)
  
        uv = surface.coordinates(n)
        normals = surface.unit_normals(uv)
        jacobians = surface.df(uv)
        k1d, k2d = np.array(surface.principal_direction(uv))

        push_forward_k1d = np.einsum('ij,ikj->ik',k1d, jacobians)
        push_forward_k2d = np.einsum('ij,ikj->ik',k2d, jacobians)

        k1, k2 = np.array(surface.principal_curvature(uv))

        #===================================
        
        # Register Surface
        ps_mesh = ps.register_surface_mesh(name, verts, faces)

        S = surface.shape_index(uv)
        S_colors = np.array([np.array(shape_scale_to_color(s)) for s in S])
        ps.get_surface_mesh(name).add_color_quantity("S_colormap", S_colors, defined_on='vertices')
        ps.get_surface_mesh(name).add_scalar_quantity("S", 
            S, defined_on='vertices', cmap='viridis')

        # Register Surface Normals
        ps.get_surface_mesh(name).add_vector_quantity(
            "normals",
            np.array(normals), 
            defined_on='vertices',
            color=web_colors["cyan"].rgb_255)

        # Register Principal Direction Vector Fields
        ps.get_surface_mesh(name).add_vector_quantity(
            "k1d",
            push_forward_k1d, 
            defined_on='vertices',
            color=web_colors["yellow"].rgb_255)

        ps.get_surface_mesh(name).add_vector_quantity(
            "k2d",
            push_forward_k2d, 
            defined_on='vertices',
            color=web_colors["red"].rgb_255)

        # Register Principal Cuvature Scalar Fields
        ps.get_surface_mesh(name).add_scalar_quantity("k1", 
            k1, defined_on='vertices', cmap='blues')

        # Register Principal Cuvature Scalar Fields
        ps.get_surface_mesh(name).add_scalar_quantity("k2", 
            k2, defined_on='vertices', cmap='reds')

        # Register Principal Cuvature Scalar Fields
        ps.get_surface_mesh(name).add_scalar_quantity("k", 
            np.abs(k1-k2), defined_on='vertices', cmap='viridis')


    ps.show()
