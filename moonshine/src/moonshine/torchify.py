import numpy as np
import tensorflow as tf
import torch


def torchify(d):
    if isinstance(d, dict):
        return {k: torchify(v) for k, v in d.items()}
    elif isinstance(d, torch.Tensor):
        return d
    elif isinstance(d, tf.Tensor):
        if d.dtype == tf.string:
            return d.numpy()  # torch doesn't suppor strings in Tensors, so just convert return as a numpy array
        else:
            return torch.from_numpy(d.numpy())
    elif isinstance(d, np.ndarray):
        if d.dtype.char in ['U', 'O']:
            return d
        else:
            return torch.from_numpy(d)
    else:
        return d
    raise NotImplementedError()
