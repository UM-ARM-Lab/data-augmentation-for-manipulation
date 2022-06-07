import torch


def pairwise_squared_distances(a, b):
    """
    Adapted from https://github.com/ClayFlannigan/icp
    Computes pairwise distances between to sets of k-dimensional points

    Args:
        a: [b, ..., n, k]
        b:  [b, ..., m, k]

    Returns: [b, ..., n, m]

    """
    a_s = a.square().sum(dim=-1, keepdim=True)  # [b, ..., n, 1]
    b_s = b.square().sum(dim=-1, keepdim=True)  # [b, ..., m, 1]
    dist = a_s - 2 * (a @ b.transpose(-1, -2)) + b_s.transpose(-2, -1)  # [b, ..., n, m]
    return dist


def pairwise_squared_distances_self(a):
    """ same as above except, specialized to one input, with the diagonal is set to inf """
    a_s = a.square().sum(dim=-1, keepdim=True)  # [b, ..., n, 1]
    d = a_s - 2 * (a @ a.transpose(-1, -2)) + a_s.transpose(-2, -1)  # [b, ..., n, n]
    n = a.shape[-2]
    batch_shape = list(a.shape[:-2])
    mask = torch.eye(n).repeat(*batch_shape, 1, 1).bool()
    d[mask] = torch.inf
    return d
