from typing import Dict
import numpy as np

import torch

from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_sequences


def vector_to_dict(description: Dict, z, device):
    start_idx = 0
    d = {}
    for k, dim in description.items():
        indices = torch.arange(start_idx, start_idx + dim).to(device)
        d[k] = torch.index_select(z, dim=-1, index=indices)
        start_idx += dim
    return d


def sequence_of_dicts_to_dict_of_tensors(seq_of_dicts, axis=0):
    dict_of_seqs = sequence_of_dicts_to_dict_of_sequences(seq_of_dicts)
    return {k: torch.stack(v, axis) for k, v in dict_of_seqs.items()}


def loss_on_dicts(loss_func, dict_true, dict_pred):
    loss_by_key = []
    for k, y_pred in dict_pred.items():
        y_true = dict_true[k]
        loss = loss_func(y_true, y_pred)
        loss_by_key.append(loss)
    return torch.mean(torch.stack(loss_by_key))


def repeat_tensor(v, repetitions, axis, new_axis):
    if np.isscalar(v):
        multiples = []
    else:
        multiples = [1] * v.ndim

    if new_axis:
        multiples.insert(axis, repetitions)
        v = v.unsqueeze(axis)
        return torch.tile(v, multiples)
    else:
        multiples[axis] *= repetitions
        return torch.tile(v, multiples)
