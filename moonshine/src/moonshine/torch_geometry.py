import torch


def pairwise_squared_distances(a, b):
    """
    Adapted from https://github.com/ClayFlannigan/icp
    Computes pairwise distances between to sets of points

    Args:
        a: [b, ..., n, k]
        b:  [b, ..., m, k]

    Returns: [b, ..., n, m]

    """
    a_s = a.square().sum(dim=-1, keepdim=True)  # [b, ..., n, 1]
    b_s = b.square().sum(dim=-1, keepdim=True)  # [b, ..., m, 1]
    dist = a_s - 2 * (a @ b.transpose(-1, -2)) + b_s.transpose(-2, -1)  # [b, ..., n, m]
    return dist
