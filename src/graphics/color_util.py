import matplotlib
import matplotlib.cm as cm
import numpy as np


def color_map(values):
    minima = np.min(values)
    maxima = np.max(values)
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap("viridis"))
    colors = [mapper.to_rgba(v) for v in values]
    return np.array(colors)[:, 0:3]
