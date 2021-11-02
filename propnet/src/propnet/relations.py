from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RelationsGraph:
    Rr_idx: torch.Tensor
    Rs_idx: torch.Tensor
    Ra: torch.Tensor


def construct_fully_connected_rel(size: int, relation_dim: int, device):
    """

    Args:
        size: number of objects/particles, $|O|$
        relation_dim: dimension of the relation vector

    Returns:

    """
    # rel is a matrix. each row is a list of indices i,j for the various objects
    rel = np.zeros((size ** 2, 2))
    size_range = np.arange(size)
    rel[:, 0] = np.repeat(size_range, size)
    rel[:, 1] = np.tile(size_range, size)

    n_rel = rel.shape[0]
    n_rel_range = np.arange(n_rel)  # [0, 1, ..., size^2]
    Rr_idx = torch.LongTensor(np.array([rel[:, 0], n_rel_range])).to(device)  # receiver indices [2, size^2]
    Rs_idx = torch.LongTensor(np.array([rel[:, 1], n_rel_range])).to(device)  # sender indices [2, size^2]
    Ra = torch.zeros(n_rel, relation_dim).to(device)  # relation attributes information

    return RelationsGraph(Rr_idx, Rs_idx, Ra)
