import tensorflow as tf


def log_barrier(x, scale, cutoff, e=0.01):
    log_cutoff = tf.math.log(scale * cutoff + e)
    return tf.maximum(-tf.math.log(scale * x + e), -log_cutoff) + log_cutoff


def exponential_barrier(x, a=1.0, b=1e-3):
    return tf.exp(-x * a + b)
