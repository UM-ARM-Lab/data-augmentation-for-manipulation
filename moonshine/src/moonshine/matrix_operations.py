from functools import lru_cache

import tensorflow as tf


def shift_and_pad(x, shift: int, pad_value: int, axis: int):
    if shift == 0:
        return x

    paddings, slices = build_shift_and_pad_params(axis, shift, x.ndim)

    padded = tf.pad(x, paddings, constant_values=pad_value)
    rolled = tf.roll(padded, shift, axis)
    shifted_and_padded = rolled[slices]
    return shifted_and_padded


@lru_cache
def build_shift_and_pad_params(axis, shift, ndim):
    paddings = []
    for i in range(ndim):
        if i == axis:
            if shift > 0:
                paddings.append((1, 0))
            else:
                paddings.append((0, 1))
        else:
            paddings.append((0, 0))
    slices = []
    for i in range(ndim):
        if i == axis:
            if shift > 0:
                slices.append(slice(1, None))
            else:
                slices.append(slice(None, -1))
        else:
            slices.append(slice(None))
    return paddings, slices


def batch_outer_product(a, b):
    """
    :param a: [batch, n]
    :param b: [batch, m]
    :return: [batch, n, m]
    """
    return tf.einsum('bn,bm->bnm', a, b)
