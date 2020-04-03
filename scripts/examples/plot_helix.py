import sys

sys.path.append("src")

from geometry.smooth.curves import Helix
from geometry.discrete.shape import PolygonSpaceCurve
from graphics.plotly_util import plot_line_segments

helix = Helix([0.0, 10.0])

path = PolygonSpaceCurve.from_space_curve(space_curve=helix, step=100)

plot_line_segments(*zip(*path.vertices))
