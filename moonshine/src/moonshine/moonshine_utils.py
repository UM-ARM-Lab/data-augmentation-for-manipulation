from typing import Dict, Optional

import numpy as np
import tensorflow as tf


def check_numerics(x, msg: Optional[str] = "found infs or nans!"):
    if isinstance(x, list):
        for v in x:
            if tf.is_tensor(v) and v.dtype in [tf.bfloat16, tf.float16, tf.float32, tf.float64]:
                tf.debugging.check_numerics(v, msg)
    elif isinstance(x, dict):
        for v in x.values():
            if tf.is_tensor(v) and v.dtype in [tf.bfloat16, tf.float16, tf.float32, tf.float64]:
                tf.debugging.check_numerics(v, msg)
    elif tf.is_tensor(x) and x.dtype in [tf.bfloat16, tf.float16, tf.float32, tf.float64]:
        tf.debugging.check_numerics(x, msg)


def numpify(x, dtype=np.float32):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, list):
        if len(x) == 0:
            return np.array(x)
        if isinstance(x[0], int):
            return np.array(x, dtype=dtype)
        elif isinstance(x[0], float):
            return np.array(x, dtype=dtype)
        elif isinstance(x[0], str):
            return np.array(x, dtype=np.str)
        else:
            l = [numpify(xi) for xi in x]
            # NOTE: if l is list of dicts for instance, we don't want to convert to an array.
            #  But if it's a list of lists (e.g. array) we do convert, so this is how we test for that
            l_arr = np.array(l)
            if l_arr.dtype in [np.float32, np.float64, np.int32, np.int64]:
                return l_arr
            else:
                return l
    elif isinstance(x, tf.Tensor):
        return x.numpy()
    elif isinstance(x, tf.Variable):
        return x.numpy()
    elif isinstance(x, dict):
        return {k: numpify(v) for k, v in x.items()}
    elif isinstance(x, tuple):
        return tuple(numpify(x_i) for x_i in x)
    elif isinstance(x, int):
        return x
    elif isinstance(x, str):
        return x
    elif isinstance(x, float):
        return x
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, np.float32):
        return x
    elif isinstance(x, np.int64):
        return x
    elif isinstance(x, np.bytes_):
        return x
    elif x is None:
        return None
    else:
        raise NotImplementedError(type(x))


def listify(x):
    def _listify(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        elif isinstance(x, tuple):
            return tuple(_listify(x_i) for x_i in x)
        elif isinstance(x, list):
            return [_listify(x_i) for x_i in x]
        elif isinstance(x, tf.Tensor):
            x_np = x.numpy()
            return _listify(x_np)
        elif isinstance(x, dict):
            return {k: _listify(v) for k, v in x.items()}
        elif isinstance(x, np.int64):
            return int(x)
        elif isinstance(x, np.int32):
            return int(x)
        elif isinstance(x, np.float64):
            return float(x)
        elif isinstance(x, np.float32):
            return float(x)
        elif isinstance(x, int):
            return x
        elif isinstance(x, float):
            return x
        elif isinstance(x, str):
            return x
        else:
            raise NotImplementedError(type(x))

    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, list):
        return [_listify(x_i) for x_i in x]
    elif isinstance(x, tf.Tensor):
        x_np = x.numpy()
        return _listify(x_np)
    elif isinstance(x, dict):
        return {k: _listify(v) for k, v in x.items()}
    elif isinstance(x, float):
        return [x]
    elif isinstance(x, int):
        return [x]
    else:
        raise NotImplementedError(type(x))


def states_are_equal(state_dict1, state_dict2):
    if state_dict1.keys() != state_dict2.keys():
        return False

    for key in state_dict1.keys():
        s1 = state_dict1[key]
        s2 = state_dict2[key]
        if not np.all(s1 == s2):
            return False

    return True


def dict_of_numpy_arrays_to_dict_of_tensors(np_dict, dtype=tf.float32):
    tf_dict = {}
    for k, v in np_dict.items():
        tf_dict[k] = tf.convert_to_tensor(v, dtype=dtype)
    return tf_dict


def dict_of_tensors_to_dict_of_numpy_arrays(tf_dict):
    np_dict = {}
    for k, v in tf_dict.items():
        np_dict[k] = v.numpy()
    return np_dict


def flatten_batch_and_time(d):
    # assumes each element in d is of shape [b, t, ...]
    return {k: tf.reshape(v, [-1] + v.shape.as_list()[2:]) for k, v in d.items()}


def sequence_of_dicts_to_dict_of_sequences(seq_of_dicts):
    # TODO: make a data structure that works both ways, as a dict and as a list
    dict_of_seqs = {}
    for d in seq_of_dicts:
        for k, v in d.items():
            if k not in dict_of_seqs:
                dict_of_seqs[k] = []
            dict_of_seqs[k].append(v)

    return dict_of_seqs


def sequence_of_dicts_to_dict_of_np_arrays(seq_of_dicts, dtype=np.float32):
    dict_of_seqs = sequence_of_dicts_to_dict_of_sequences(seq_of_dicts)
    return {k: np.array(v, dtype=dtype) for k, v in dict_of_seqs.items()}


def sequence_of_dicts_to_dict_of_tensors(seq_of_dicts, axis=0):
    dict_of_seqs = sequence_of_dicts_to_dict_of_sequences(seq_of_dicts)
    return {k: tf.stack(v, axis) for k, v in dict_of_seqs.items()}


def repeat(d: Dict, repetitions: int, axis: int, new_axis: bool):
    out_d = {}
    for k, v in d.items():
        if np.isscalar(v):
            multiples = []
        else:
            multiples = [1] * v.ndim
        if new_axis:
            multiples.insert(axis, repetitions)
            v = tf.expand_dims(v, axis=axis)
            out_d[k] = tf.tile(v, multiples)
        else:
            multiples[axis] *= repetitions
            out_d[k] = tf.tile(v, multiples)
    return out_d


def dict_of_sequences_to_sequence_of_dicts_tf(dict_of_seqs, time_axis=0):
    # FIXME: a common problem I have is that I have a dictionary of tensors, each with the same shape in the first M dimensions
    # and I want to get those shapes, but I don't care which key/value I use. Feels like I need a different datastructure here.
    seq_of_dicts = []
    # assumes all values in the dict have the same first dimension size (num time steps)
    T = list(dict_of_seqs.values())[0].shape[time_axis]
    for t in range(T):
        dict_t = {}
        for k, v in dict_of_seqs.items():
            dict_t[k] = tf.gather(v, t, axis=time_axis)
        seq_of_dicts.append(dict_t)

    return seq_of_dicts


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


def remove_batch(*xs):
    if len(xs) == 1:
        return remove_batch_single(xs[0])
    else:
        return [remove_batch_single(x) for x in xs]


def add_time_dim(*xs, batch_axis=1):
    if len(xs) == 1:
        return add_batch_single(xs[0], batch_axis)
    else:
        return [add_batch_single(x, batch_axis) for x in xs]


def add_batch(*xs, batch_axis=0):
    if len(xs) == 1:
        return add_batch_single(xs[0], batch_axis)
    else:
        return [add_batch_single(x, batch_axis) for x in xs]


def remove_batch_single(x):
    if isinstance(x, dict):
        return {k: remove_batch_single(v) for k, v in x.items()}
    elif isinstance(x, int):
        return x
    elif isinstance(x, float):
        return x
    elif isinstance(x, tf.Tensor):
        x.is_batched = False
        if len(x.shape) == 0:
            return x
        else:
            return x[0]
    else:
        return x[0]


def add_batch_single(x, batch_axis=0):
    if isinstance(x, np.ndarray):
        return np.expand_dims(x, axis=batch_axis)
    elif isinstance(x, list) and isinstance(x[0], dict):
        return [(add_batch_single(v)) for v in x]
    elif isinstance(x, tf.Tensor):
        x = tf.expand_dims(x, axis=batch_axis)
        x.is_batched = True
        return x
    elif isinstance(x, dict):
        return {k: add_batch_single(v, batch_axis) for k, v in x.items()}
    else:
        return np.array([x])


def gather_dict(d: Dict, indices, axis: int = 0):
    """
    :param d: a dictionary where each value is a tensor/array with the same dimension along 'axis'
    :param indices: a 1-d tensor/array/vector of ints describing which elements to include from d
    :param axis: the axis to gather along
    :return:
    """
    return {k: tf.gather(v, indices, axis=axis) for k, v in d.items()}


def vector_to_dict(description: Dict, z):
    start_idx = 0
    d = {}
    for k, dim in description.items():
        indices = tf.range(start_idx, start_idx + dim)
        d[k] = tf.gather(z, indices, axis=-1)
        start_idx += dim
    return d


def flatten_after(x, axis: int = 0):
    """ [N1, N2, ...] -> [N1, ..., N[axis], -1] """
    new_shape = x.shape.as_list()[:axis + 1] + [-1]
    return tf.reshape(x, new_shape)
