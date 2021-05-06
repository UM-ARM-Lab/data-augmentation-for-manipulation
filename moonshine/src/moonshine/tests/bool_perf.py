import tensorflow as tf
from time import perf_counter

from moonshine.simple_profiler import SimpleProfiler


def float32_involved(a, b, c):
    return a * (1 - b) * (1 - c)


def boolean_involved(a, b, c):
    return tf.cast(tf.reduce_all([tf.cast(a, tf.bool),
                                  tf.logical_not(tf.cast(b, tf.bool)),
                                  tf.logical_not(tf.cast(c, tf.bool))]), tf.float32)


def float32_and(a, b):
    return a * b


def boolean_and(a, b):
    return tf.cast(tf.logical_and(tf.cast(a, tf.bool), tf.cast(b, tf.bool)), tf.float32)


def main():
    s = 100
    a = tf.random.uniform([s, s, s], dtype=tf.float32)
    b = tf.random.uniform([s, s, s], dtype=tf.float32)
    c = tf.random.uniform([s, s, s], dtype=tf.float32)
    n = 50000

    p = SimpleProfiler()
    # p.profile(n, boolean_and, a, b)
    p.profile(n, boolean_involved, a, b, c)
    print("bool:")
    print(p)

    # p.profile(n, float32_and, a, b)
    p.profile(n, float32_involved, a, b, c)
    print("float:")
    print(p)


if __name__ == '__main__':
    main()
