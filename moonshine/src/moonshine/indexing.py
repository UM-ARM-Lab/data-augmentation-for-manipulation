from typing import Dict, List
import numpy as np

import tensorflow as tf

from moonshine.moonshine_utils import numpify, remove_batch, add_batch


# TODO: make all these indexing functions that take lists of keys have the "inclusive" argument
def index_batch_time(example: Dict, keys, b: int, t: int):
    e_t = {k: example[k][b, t] for k in keys}
    return e_t


def index_batch_time_with_metadata(metadata: Dict, example: Dict, keys, b: int, t: int):
    e_t = {k: example[k][b, t] for k in keys}
    e_t.update(metadata)
    return e_t


def index_time_with_metadata(metadata: Dict, example: Dict, keys, t: int):
    e_t = {k: index_time_kv(k, example[k], t) for k in keys}
    e_t.update(metadata)
    return e_t


def index_time_np(e: Dict, time_indexed_keys: List[str], t: int, inclusive: bool):
    return numpify(index_time(e, time_indexed_keys, t, inclusive=inclusive))


def index_time(e: Dict, time_indexed_keys: List[str], t: int, inclusive: bool):
    return remove_batch(index_time_batched(add_batch(e), time_indexed_keys, t, inclusive=inclusive))


def index_time_batched(e: Dict, time_indexed_keys: List[str], t: int, inclusive: bool):
    """
    Args:
        e:
        time_indexed_keys: keys for which the indexing [:, t] should be performed
        t:
        inclusive: If true, any key not listed in time_indexed_keys will still be included as-is in the output

    Returns:

    """
    e_t = {}
    if inclusive:
        for k, v in e.items():
            e_t[k] = index_time_batched_kv_if_time_indexed(k, v, t, time_indexed_keys)
    else:
        for k in time_indexed_keys:
            v = e[k]
            e_t[k] = index_time_batched_kv_if_time_indexed(k, v, t, time_indexed_keys)
    return e_t


def index_time_batched_kv_if_time_indexed(k, v, t, time_indexed_keys):
    if k in time_indexed_keys:
        if v.ndim == 1:
            return v
        elif t < v.shape[1]:
            return v[:, t]
        elif t == v.shape[1]:
            return v[:, t - 1]
        else:
            err_msg = f"time index {t} out of bounds for {k} which has shape {v.shape}"
            raise IndexError(err_msg)
    else:
        return v


def index_state_action_with_metadata(metadata: Dict,
                                     example: Dict,
                                     state_keys: List[str],
                                     action_keys: List[str], t: int):
    s_t = {}
    a_t = {}

    # just pick the first key for this check
    test_k = action_keys[0]
    test_v = example[test_k]
    if t < test_v.shape[0]:
        for state_key in state_keys:
            s_t[state_key] = example[state_key][t]
        for action_key in action_keys:
            a_t[action_key] = example[action_key][t]
    elif t == test_v.shape[0]:
        for state_key in state_keys:
            s_t[state_key] = example[state_key][t - 1]
        for action_key in action_keys:
            a_t[action_key] = example[action_key][t - 1]
    else:
        err_msg = f"time index {t} out of bounds for {test_k} which has shape {test_v.shape}"
        raise IndexError(err_msg)

    s_t.update(metadata)

    return s_t, a_t


# TODO: fix API
def index_time_2(example: Dict, k: str, t):
    v = example[k]
    return index_time_kv(k, v, t)


def index_time_kv(k, v, t):
    if t < v.shape[0]:
        return v[t]
    elif t == v.shape[0]:
        return v[t - 1]
    else:
        err_msg = f"time index {t} out of bounds for {k} which has shape {v.shape}"
        raise IndexError(err_msg)


def index_time_batched_kv(k, v, t):
    if v.ndim == 1:
        return v
    elif t < v.shape[1]:
        return v[:, t]
    elif t == v.shape[1]:
        return v[:, t - 1]
    else:
        err_msg = f"time index {t} out of bounds for {k} which has shape {v.shape}"
        raise IndexError(err_msg)


def index_label_time_batched(example: Dict, t: int):
    if t == 0:
        # it makes no sense to have a label at t=0, labels are for transitions/sequences
        # the plotting function that consumes this should use None correctly
        return None
    return example['is_close'][:, t]


def index_dict_of_batched_tensors_np(in_dict: Dict, index: int, batch_axis: int = 0):
    out_dict_tf = index_dict_of_batched_tensors_tf(in_dict=in_dict, index=index, batch_axis=batch_axis)
    return {k: v.numpy() for k, v in out_dict_tf.items()}


def index_dict_of_batched_tensors_tf(in_dict: Dict, index: int, batch_axis: int = 0, keep_dims=False):
    out_dict = {}
    for k, v in in_dict.items():
        try:
            v_i = tf.gather(v, index, axis=batch_axis)
            if keep_dims:
                v_i = tf.expand_dims(v_i, axis=batch_axis)
            out_dict[k] = v_i
        except ValueError:
            try:
                v_i = np.take(v, index, axis=batch_axis)
                if keep_dims:
                    v_i = np.expand_dims(v_i, axis=batch_axis)
                out_dict[k] = v_i
            except ValueError:
                pass
    return out_dict
