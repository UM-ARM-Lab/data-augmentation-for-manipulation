from typing import Dict

import numpy as np
import torch

from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_sequences
from moonshine.torchify import torchify


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
    out_dict = {}
    for k, v in dict_of_seqs.items():
        torch_v = [torchify(v_t) for v_t in v]
        try:
            out_dict[k] = torch.stack(torch_v, axis)
        except TypeError:
            out_dict[k] = torch_v
    return out_dict


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
