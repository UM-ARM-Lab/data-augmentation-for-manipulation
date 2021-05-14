from unittest import TestCase
import numpy as np

import tensorflow as tf

from moonshine.geometry import make_rotation_matrix_like, rotate_points_3d
from moonshine.matrix_operations import batch_outer_product
from moonshine.tests.testing_utils import assert_close_tf


class Test(TestCase):
    def test_rotate_points_3d(self):
        theta = tf.constant([0.0, np.pi, np.pi / 4], tf.float32)
        test_points = tf.constant([[0.1, 0, 0], [0, 0.1, 0], [0.1, 0, 0.1]], tf.float32)
        rotation_matrix = make_rotation_matrix_like(test_points, theta)
        rotated_points = rotate_points_3d(rotation_matrix, test_points)
        expected_points = tf.constant([[0.1, 0, 0], [0, -0.1, 0], [0.07071067, 0.07071067, 0.1]], dtype=tf.float32)
        assert_close_tf(rotated_points, expected_points)
