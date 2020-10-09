import sys

sys.path.append("src")

from geometry.smooth import curves
from geometry.discrete.shape import PolygonSpaceCurve
import numpy as np
import polyscope as ps

ps.init()

curve = curves.Line3D([1,1,1],[2,4,5])
path = PolygonSpaceCurve.from_space_curve(space_curve=curve, step=100)
ps_net = ps.register_curve_network("helix", path.vertices, "line")

ps.show()