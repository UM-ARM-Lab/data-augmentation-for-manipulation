import tensorflow as tf


def log_barrier(x, a=1.0, b=1e-3):
    return -tf.math.log(x * a + b)
