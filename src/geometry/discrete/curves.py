from geometry.smooth.curve import SpaceCurve
from util import sample_range

class PolygonSpaceCurve(object):
    """ A connected series of line segment in 3D space

    Also known as path or linestring
	"""

    def __init__(self, vertices):
        self.vertices = vertices

    @classmethod
    def from_space_curve(cls, space_curve: SpaceCurve, step: float):
        """ linearly rasterize a smooth SpaceCurve into a PolygonSpaceCurve
		"""
        coords = sample_range(space_curve.t_range(), step)
        vertices = space_curve.f(coords)
        return cls(vertices)