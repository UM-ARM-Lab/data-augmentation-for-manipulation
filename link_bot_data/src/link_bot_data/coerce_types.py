from typing import Dict

import numpy as np
import tensorflow as tf
import torch

import genpy


def coerce_types(d: Dict):
    """
    Converts the types of things in the dict to whatever we want for saving it
    Args:
        d:

    Returns:

    """
    out_d = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            if v.dtype == np.float64:
                out_d[k] = v.astype(np.float32)
            else:
                out_d[k] = v
        elif isinstance(v, np.float64):
            out_d[k] = np.float32(v)
        elif isinstance(v, np.float32):
            out_d[k] = v
        elif isinstance(v, np.int64):
            out_d[k] = v
        elif isinstance(v, tf.Tensor):
            v = v.numpy()
            if isinstance(v, np.ndarray):
                if v.dtype == np.float64:
                    out_d[k] = v.astype(np.float32)
                else:
                    out_d[k] = v
            elif isinstance(v, np.float64):
                out_d[k] = np.float32(v)
            else:
                out_d[k] = v
        elif isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
            if isinstance(v, np.ndarray):
                if v.dtype == np.float64:
                    out_d[k] = v.astype(np.float32)
                else:
                    out_d[k] = v
            elif isinstance(v, np.float64):
                out_d[k] = np.float32(v)
            else:
                out_d[k] = v
        elif isinstance(v, genpy.Message):
            out_d[k] = v
        elif isinstance(v, str):
            out_d[k] = v
        elif isinstance(v, bytes):
            out_d[k] = v
        elif isinstance(v, int):
            out_d[k] = v
        elif isinstance(v, float):
            out_d[k] = v
        elif isinstance(v, list):
            v0 = v[0]
            if isinstance(v0, int):
                out_d[k] = np.array(v)
            elif isinstance(v0, float):
                out_d[k] = np.array(v, dtype=np.float32)
            elif isinstance(v0, genpy.Message):
                out_d[k] = np.array(v, dtype=object)
            elif isinstance(v0, str):
                out_d[k] = v
            elif isinstance(v0, list):
                v00 = v0[0]
                if isinstance(v00, int):
                    out_d[k] = np.array(v)
                elif isinstance(v00, float):
                    out_d[k] = np.array(v, dtype=np.float32)
                elif isinstance(v00, str):
                    out_d[k] = v
                elif isinstance(v00, genpy.Message):
                    out_d[k] = np.array(v, dtype=object)
            elif isinstance(v0, np.ndarray):
                if v0.dtype == np.float64:
                    out_d[k] = np.array(v).astype(np.float32)
                else:
                    out_d[k] = np.array(v)
            elif isinstance(v0, tf.Tensor):
                out_d[k] = tf.convert_to_tensor(v)
            else:
                raise NotImplementedError(f"{k} {type(v)} {v}")
        elif isinstance(v, dict):
            out_d[k] = coerce_types(v)
        else:
            raise NotImplementedError(f"{k} {type(v)}")
    assert len(out_d) == len(d)
    return out_d