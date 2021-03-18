#!/usr/bin/env python
import unittest

import tensorflow as tf

from moonshine.gpu_config import limit_gpu_mem
from moonshine.image_augmentation import random_flip
from moonshine.tests.testing_utils import assert_close_tf

limit_gpu_mem(0.1)


class Test(unittest.TestCase):
    def test_random_flip(self):
        x = tf.constant([[1, 0, 0], [0, 1, 1]], dtype=tf.float32)
        y = random_flip(x, p_flip=0.0)
        assert_close_tf(x, y)

        x = tf.constant([[1, 0, 0], [0, 1, 1]], dtype=tf.float32)
        y = random_flip(x, p_flip=1.0)
        assert_close_tf(x, 1 - y)

        x = tf.constant([[1, 0, 0], [0, 1, 1]], dtype=tf.float32)
        y = random_flip(x, p_flip=0.5)
        self.assertEqual(tf.math.reduce_min(y).numpy(), 0)
        self.assertEqual(tf.math.reduce_max(y).numpy(), 1)


if __name__ == '__main__':
    unittest.main()
