import numpy as np


def sample_range(start_end, step):
    start, end = start_end
    return np.linspace(start, end, step)
