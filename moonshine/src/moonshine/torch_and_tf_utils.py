from typing import Dict

import numpy as np
import tensorflow as tf
import torch

from moonshine.tensorflow_utils import repeat_tensor


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


def add_batch(*xs, batch_axis=0, keys=None):
    if len(xs) == 1:
        return add_batch_single(xs[0], batch_axis, keys=None)
    else:
        return [add_batch_single(x, batch_axis, keys=None) for x in xs]


def remove_batch_single(x):
    if isinstance(x, dict):
        return {k: remove_batch_single(v) for k, v in x.items()}
    elif isinstance(x, int):
        return x
    elif isinstance(x, float):
        return x
    elif isinstance(x, tf.Tensor):
        if len(x.shape) == 0:
            return x
        elif x.shape[0] == 0:
            return tf.reshape(x, [0] + x.shape[2:])
        else:
            return x[0]
    else:
        return x[0]


def add_batch_single(x, batch_axis=0, keys=None):
    if isinstance(x, np.ndarray):
        return np.expand_dims(x, axis=batch_axis)
    elif isinstance(x, list) and isinstance(x[0], dict):
        return [(add_batch_single(v)) for v in x]
    elif isinstance(x, torch.Tensor):
        x = torch.unsqueeze(x, dim=batch_axis)
        return x
    elif isinstance(x, tf.Tensor):
        x = tf.expand_dims(x, axis=batch_axis)
        return x
    elif isinstance(x, tf.Variable):
        x = tf.expand_dims(x, axis=batch_axis)
        return x
    elif isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if keys is not None and k in keys:
                out[k] = add_batch_single(v, batch_axis)
            elif keys is not None and k not in keys:
                out[k] = v
            elif keys is None:
                out[k] = add_batch_single(v, batch_axis)
        return out
    else:
        return np.array([x])


def repeat(d: Dict, repetitions: int, axis: int, new_axis: bool):
    return {k: repeat_tensor(v, repetitions, axis, new_axis) for k, v in d.items()}
