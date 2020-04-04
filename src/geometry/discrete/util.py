import numpy as np


def get_faces(ulist, vlist):
    """ Convert two coordinate arrays into a triangulation
    of the plane

    Parameters
    ----------
    ulist: [float]
    vlist: [float]

    Returns
    -------
    nd.array N x 3
        A list of faces represented by three indices into the list
        of coordinates produced by `get_verts(ulist, vlist)`
    """
    width = len(ulist)
    faces = []
    for i in range(len(ulist) - 1):
        for j in range(len(vlist) - 1):
            topleft = j * width + i
            topright = topleft + 1
            bottomleft = ((j + 1) * width) + i
            bottomright = bottomleft + 1
            one = [topleft, topright, bottomleft]
            two = [bottomleft, topright, bottomright]
            faces.append(one)
            faces.append(two)

    return faces


def get_verts(ulist, vlist, func):
    """ Convert two coordinate arrays into a list of coordinates
    """
    verts = []
    for u in ulist:
        for v in vlist:
            verts.append(func(u, v))
    return verts


def triangulate_surface(surface, step):
    ulist = surface.u_linspace()
    vlist = surface.v_linspace()
    coords = []
    for u in ulist:
        for v in vlist:
            coords.append([u, v])

    vertices = surface.f(np.array(coords))

    face_indices = get_faces(ulist, vlist)

    return vertices, face_indices
