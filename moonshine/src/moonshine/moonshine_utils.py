import multiprocessing
from typing import List

import numpy as np


def get_common_dicts(examples):
    e_common = {}
    keys = set()
    for k in examples[0].keys():
        k_is_common = True
        for e in examples[1:]:
            if k not in e:
                k_is_common = False
        if k_is_common:
            keys.add(k)
    for k in keys:
        # examples[0] must have k, because k is in all examples
        e_common[k] = examples[0][k]

    return e_common


def states_are_equal(state_dict1, state_dict2):
    if state_dict1.keys() != state_dict2.keys():
        return False

    for key in state_dict1.keys():
        s1 = state_dict1[key]
        s2 = state_dict2[key]
        if not np.all(s1 == s2):
            return False

    return True


def sequence_of_dicts_to_dict_of_sequences(seq_of_dicts):
    # TODO: make a data structure that works both ways, as a dict and as a list
    dict_of_seqs = {}
    for d in seq_of_dicts:
        for k, v in d.items():
            if k not in dict_of_seqs:
                dict_of_seqs[k] = []
            dict_of_seqs[k].append(v)

    return dict_of_seqs


def sequence_of_dicts_to_dict_of_np_arrays(seq_of_dicts, axis):
    dict_of_seqs = sequence_of_dicts_to_dict_of_sequences(seq_of_dicts)
    return {k: np.stack(v, axis) for k, v in dict_of_seqs.items()}


def dict_of_sequences_to_sequence_of_dicts(dict_of_seqs, time_axis=0):
    seq_of_dicts = []
    # assumes all values in the dict have the same first dimension size (num time steps)
    T = len(list(dict_of_seqs.values())[time_axis])
    for t in range(T):
        dict_t = {}
        for k, v in dict_of_seqs.items():
            dict_t[k] = np.take(v, t, axis=time_axis)
        seq_of_dicts.append(dict_t)

    return seq_of_dicts


def list_of_tuples_to_tuple_of_lists(values: List[tuple]):
    tuple_size = len(values[0])
    lists = []
    for i in range(tuple_size):
        lists.append([v[i] for v in values])
    return tuple(lists)


def get_num_workers(batch_size):
    return min(batch_size, multiprocessing.cpu_count())
