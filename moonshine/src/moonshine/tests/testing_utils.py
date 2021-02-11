from typing import Dict

import numpy as np
import tensorflow as tf


def are_dicts_close_np(a: Dict, b: Dict):
    assert (set(a.keys()) == set(b.keys()))
    for k, v1 in a.items():
        v2 = b[k]
        if not np.allclose(v1, v2):
            return False
    return True


def assert_dicts_close_np(a, b):
    for v1, v2 in zip(a.values(), b.values()):
        assert np.allclose(v1, v2)


def is_close_tf(a, b, rtol=1e-4, atol=1e-8):
    return tf.reduce_all(tf.abs(a - b) <= tf.abs(b) * rtol + atol)


def assert_close_tf(a, b, rtol=1e-4, atol=1e-8):
    assert is_close_tf(a, b, rtol=rtol, atol=atol)


def assert_dicts_close_tf(a, b, rtol=1e-4, atol=1e-8):
    for k, v1, v2 in zip(a.keys(), a.values(), b.values()):
        assert tf.reduce_all(tf.abs(v1 - v2) <= tf.abs(v2) * rtol + atol), f"key {k}: {v1} vs {v2}"


def assert_list_of_dicts_close_np(a, b):
    for a_i, b_i in zip(a, b):
        assert_dicts_close_np(a_i, b_i)


def assert_list_of_dicts_close_tf(a, b, rtol=1e-4, atol=1e-8):
    for a_i, b_i in zip(a, b):
        assert_dicts_close_tf(a_i, b_i, rtol, atol)
