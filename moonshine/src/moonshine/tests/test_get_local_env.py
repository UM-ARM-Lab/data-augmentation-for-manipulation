#!/usr/bin/env python
import unittest

import numpy as np
import tensorflow as tf

from moonshine.get_local_environment_tf import get_local_env_and_origin_2d_tf, get_local_env_and_origin_3d_tf_old
from moonshine.gpu_config import limit_gpu_mem
from moonshine.tests.testing_utils import assert_close_tf

limit_gpu_mem(0.1)


class Test(unittest.TestCase):
    def test_get_local_env_and_origin_2d(self):
        res = [0.01, 0.01]
        full_h_rows = 5
        full_w_cols = 5
        local_h_rows = 3
        local_w_cols = 3

        center_point = np.array([[0, 0],
                                 [-0.01, 0.01]], np.float32)

        full_env = np.zeros([2, full_h_rows, full_w_cols], dtype=np.float32)
        full_env_origin = np.array([[full_h_rows / 2, full_w_cols / 2], [full_h_rows / 2, full_w_cols / 2]], dtype=np.float32)
        full_env[:, 1, 2] = 1
        full_env[:, 2, 3] = 1

        local_env, local_env_origin = get_local_env_and_origin_2d_tf(center_point,
                                                                     full_env,
                                                                     full_env_origin,
                                                                     res,
                                                                     local_h_rows,
                                                                     local_w_cols)
        expected_origins = tf.constant([[1.5, 1.5], [0.5, 2.5]], dtype=tf.float32)
        expected_env = tf.constant(
            [[[0, 1, 0],
              [0, 0, 1],
              [0, 0, 0]],
             [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]]],
            dtype=tf.float32)
        assert_close_tf(local_env_origin, expected_origins)
        assert_close_tf(local_env, expected_env)

    def test_get_local_env_and_origin_3d(self):
        res = [0.01, 0.01]
        full_h_rows = 5
        full_w_cols = 5
        full_c_channels = 5
        local_h_rows = 3
        local_w_cols = 3
        local_c_channels = 3

        center_point = np.array([[0, 0, 0],
                                 [-0.01, 0.01, 0.01]], np.float32)

        full_env = np.zeros([2, full_h_rows, full_w_cols, full_c_channels], dtype=np.float32)
        full_env_origin = np.array([[full_h_rows / 2, full_w_cols / 2, full_c_channels / 2],
                                    [full_h_rows / 2, full_w_cols / 2, full_c_channels / 2]],
                                   dtype=np.float32)
        full_env[:, 1, 2, 1] = 1
        full_env[:, 2, 3, 3] = 1

        local_env, local_env_origin = get_local_env_and_origin_3d_tf_old(center_point,
                                                                         full_env,
                                                                         full_env_origin,
                                                                         res,
                                                                         local_h_rows,
                                                                         local_w_cols,
                                                                         local_c_channels)
        expected_origins = tf.constant([[1.5, 1.5, 1.5], [0.5, 2.5, 0.5]], dtype=tf.float32)
        expected_env_occupied_indicies= tf.constant(
            [
                [0, 0, 1, 0],
                [0, 1, 2, 2],
            ],
            dtype=tf.float32)
        assert_close_tf(local_env_origin, expected_origins)
        assert_close_tf(tf.cast(tf.where(local_env > 0.5), tf.float32), expected_env_occupied_indicies)
