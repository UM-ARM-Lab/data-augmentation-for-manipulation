from unittest import TestCase

import tensorflow as tf

from moonshine.matrix_operations import batch_outer_product, shift_and_pad
from moonshine.tests.testing_utils import assert_close_tf


class Test(TestCase):
    def test_batch_outer_product(self):
        a = tf.constant([[1, 2], [0, 2]], dtype=tf.float32)
        b = tf.constant([[3, 4, 5], [1, 4, 2]], dtype=tf.float32)
        out = batch_outer_product(a, b)
        expected = tf.constant([[[3, 4, 5], [6, 8, 10]], [[0, 0, 0], [2, 8, 4]]], dtype=tf.float32)
        assert_close_tf(out, expected)

    def test_shift_and_pad(self):
        p = -999
        x = tf.constant([[1, 2], [0, 2]], dtype=tf.float32)
        expected = tf.constant([[p, 1], [p, 0]], dtype=tf.float32)
        out = shift_and_pad(x, shift=1, pad_value=p, axis=1)
        assert_close_tf(out, expected)

        expected = tf.constant([[2, p], [2, p]], dtype=tf.float32)
        out = shift_and_pad(x, shift=-1, pad_value=p, axis=1)
        assert_close_tf(out, expected)

        expected = tf.constant([[p, p], [1, 2]], dtype=tf.float32)
        out = shift_and_pad(x, shift=1, pad_value=p, axis=0)
        assert_close_tf(out, expected)

        expected = tf.constant([[0, 2], [p, p]], dtype=tf.float32)
        out = shift_and_pad(x, shift=-1, pad_value=p, axis=0)
        assert_close_tf(out, expected)
