import numpy as np


def homogeneous(points):
    return np.concatenate([points, np.ones_like(points[..., 0:1])], axis=-1)
