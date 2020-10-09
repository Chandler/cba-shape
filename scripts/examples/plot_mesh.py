import sys

sys.path.append("src")

from geometry.smooth import curves
import numpy as np
import polyscope as ps
from thirdparty.octasphere import octasphere


ps.init()

verts, faces = octasphere(0, 1)

ps_mesh = ps.register_surface_mesh("name", verts, faces)

ps.show()