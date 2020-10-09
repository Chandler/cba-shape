import matplotlib
import matplotlib.cm as cm
import numpy as np
from jax import np as jnp
import math

class WebColor(object):
    def __init__(self, rgb_tuple):
        #(r,g,b) 0-255
        self.rgb_tuple = rgb_tuple

    @property
    def rgb_255(self):
        return self.rgb_tuple

    @property
    def rgb_01(self):
        return np.divide(self.rgb_tuple, 255.0)

    @property
    def rgba_01(self, a=0):
        return np.divide(list(self.rgb_tuple) + [a], 255.0)
    
web_colors = {
    "cyan": WebColor((20, 211, 224)),
    "yellow": WebColor((252, 186, 3)),
    "red": WebColor((222, 37, 9)),
    "pink": WebColor((227, 154, 221)),
}

def color_map(values, map_name="RdYlGn"):
    """ Map a range of values to colors using a matplotlib
    color map
    """
    minima = np.min(values)
    maxima = np.max(values)
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(map_name))
    colors = [mapper.to_rgba(v) for v in values]
    return np.array(colors)[:, 0:3]

# http://jcgt.org/published/0002/02/01/paper.pdf
# Simple Analytic (Differentiable!) Approximations to the CIE XYZ
# Color Matching Functions
def CIEXfitSimple(wave):
    t1 = (wave-595.8)/33.33
    t2 = (wave-446.88)/19.44
    return 1.065 * jnp.exp(-0.5*t1*t1) + 0.366 * jnp.exp(-0.5*t2*t2)

def CIEYfitSimple(wave):
    t1 = (jnp.log(wave)-jnp.log(556.3))/0.075
    return 1.014 * jnp.exp(-0.5*t1*t1)

def CIEZfitSimple(wave):
    t1 = (jnp.log(wave)-jnp.log(449.8))/0.051
    return 1.839 * jnp.exp(-0.5*t1*t1)

sRGB_primaries = \
    jnp.array([
        [ 0.41231515, 0.2126, 0.01932727],
        [ 0.3576    , 0.7152, 0.1192    ],
        [ 0.1805    , 0.0722, 0.95063333]])
       
def xyz_to_srgb(xyz):
    primary_inverse = jnp.linalg.inv(sRGB_primaries)
    return xyz.T.dot(primary_inverse)

# visible wavelengths of light
visual_range_nm = [400.0, 700.0]