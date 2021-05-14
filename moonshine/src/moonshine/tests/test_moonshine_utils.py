from unittest import TestCase

import numpy as np
import tensorflow as tf

from moonshine.moonshine_utils import dict_of_sequences_to_sequence_of_dicts, dict_of_sequences_to_sequence_of_dicts_tf, \
    flatten_batch_and_time, gather_dict, repeat, add_time_dim, swap_xy
from moonshine.tests import testing_utils


class Test(TestCase):
    def test_dict_of_sequences_to_sequence_of_list_of_dicts(self):
        d = {'a': [1, 2, 3]}
        expected = [{'a': 1}, {'a': 2}, {'a': 3}]
        out = dict_of_sequences_to_sequence_of_dicts(d, time_axis=0)
        testing_utils.assert_list_of_dicts_close_np(out, expected)

    def test_dict_of_sequences_to_sequence_of_list_of_dicts_time_dim(self):
        d = {'a': np.array([[1, 2, 3], [4, 5, 6]])}
        expected = [{'a': np.array([1, 4])}, {'a': np.array([2, 5])}, {'a': np.array([3, 6])}]
        out = dict_of_sequences_to_sequence_of_dicts_tf(d, time_axis=1)
        testing_utils.assert_list_of_dicts_close_np(out, expected)

    def test_dict_of_sequences_to_sequence_of_list_of_dicts_tf(self):
        d = {'a': tf.constant([1, 2, 3], tf.float32)}
        expected = [{'a': tf.constant([1], tf.float32)},
                    {'a': tf.constant([2], tf.float32)},
                    {'a': tf.constant([3], tf.float32)}]
        out = dict_of_sequences_to_sequence_of_dicts_tf(d, time_axis=0)
        testing_utils.assert_list_of_dicts_close_tf(out, expected)

    def test_dict_of_sequences_to_sequence_of_list_of_dicts_tf_time_dim(self):
        d = {'a': tf.constant([[1, 2, 3], [4, 5, 6]], tf.float32)}
        expected = [{'a': tf.constant([1, 4], tf.float32)},
                    {'a': tf.constant([2, 5], tf.float32)},
                    {'a': tf.constant([3, 6], tf.float32)}]
        out = dict_of_sequences_to_sequence_of_dicts_tf(d, time_axis=1)
        testing_utils.assert_list_of_dicts_close_tf(out, expected)

    def test_flatten_batch_and_time(self):
        in_x = tf.constant([[[1, 2], [2, 2]], [[3, 2], [4, 2]], [[5, 2], [6, 2]]], dtype=tf.float32)
        expected_x = tf.constant([[1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2]], dtype=tf.float32)
        input_dict = {
            'x': in_x
        }
        expected_dict = {
            'x': expected_x
        }
        out_dict = flatten_batch_and_time(input_dict)
        testing_utils.assert_dicts_close_tf(out_dict, expected_dict)

    def test_slice_dict(self):
        input_dict = {
            'a': tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=tf.float32),
            'b': tf.constant([[0, 2], [3, 5], [6, 8]], dtype=tf.float32),
            'c': tf.constant([0, 5, 7], dtype=tf.float32),
        }
        expected_dict = {
            'a': tf.constant([[0, 1, 2], [6, 7, 8]], dtype=tf.float32),
            'b': tf.constant([[0, 2], [6, 8]], dtype=tf.float32),
            'c': tf.constant([0, 7], dtype=tf.float32),
        }
        out_dict = gather_dict(input_dict, [0, 2])
        testing_utils.assert_dicts_close_tf(out_dict, expected_dict)

    def test_repeat(self):
        input_dict = {
            'a': tf.constant([[2], [5], [8]], dtype=tf.float32),
            'b': tf.constant([0, 1], dtype=tf.float32),
        }

        ########################################################################
        expected_dict = {
            'a': tf.constant([[[2], [5], [8]], [[2], [5], [8]]], dtype=tf.float32),
            'b': tf.constant([[0, 1], [0, 1]], dtype=tf.float32),
        }
        out_dict = repeat(input_dict, repetitions=2, axis=0, new_axis=True)
        testing_utils.assert_dicts_close_tf(out_dict, expected_dict)

        ########################################################################
        expected_dict = {
            'a': tf.constant([[2], [5], [8], [2], [5], [8]], dtype=tf.float32),
            'b': tf.constant([0, 1, 0, 1], dtype=tf.float32),
        }
        out_dict = repeat(input_dict, repetitions=2, axis=0, new_axis=False)
        testing_utils.assert_dicts_close_tf(out_dict, expected_dict)

        ########################################################################
        expected_dict = {
            'a': tf.constant([[[2], [2]], [[5], [5]], [[8], [8]]], dtype=tf.float32),
            'b': tf.constant([[0, 0], [1, 1]], dtype=tf.float32),
        }
        out_dict = repeat(input_dict, repetitions=2, axis=1, new_axis=True)
        testing_utils.assert_dicts_close_tf(out_dict, expected_dict)

        ########################################################################
        with self.assertRaises(IndexError):
            # you can't ask for an axis that doesn't exist without new_axis=True
            out_dict = repeat(input_dict, repetitions=2, axis=1, new_axis=False)

    def test_add_time_dim(self):
        input_dict = {
            'a': tf.constant([[2], [5], [8]], dtype=tf.float32),
            'b': tf.constant([0, 1], dtype=tf.float32),
        }

        ########################################################################
        expected_dict = {
            'a': tf.constant([[[2]], [[5]], [[8]]], dtype=tf.float32),
            'b': tf.constant([[0], [1]], dtype=tf.float32),
        }
        out_dict = add_time_dim(input_dict)
        testing_utils.assert_dicts_close_tf(out_dict, expected_dict)

    def test_swap_xy(self):
        a = tf.constant([0.0, 0.1, 0.2])
        b = tf.constant([0.1, 0.0, 0.2])
        testing_utils.assert_close_tf(swap_xy(a), b)

        a = tf.constant([[0.0, 0.1, 0.2]])
        b = tf.constant([[0.1, 0.0, 0.2]])
        testing_utils.assert_close_tf(swap_xy(a), b)

        a = tf.constant([[0.0, 0.1, 0.2], [0.1, 0.2, 0.3]])
        b = tf.constant([[0.1, 0.0, 0.2], [0.2, 0.1, 0.3]])
        testing_utils.assert_close_tf(swap_xy(a), b)

        a = tf.constant([[[0.0, 0.1, 0.2], [0.1, 0.2, 0.3]]])
        b = tf.constant([[[0.1, 0.0, 0.2], [0.2, 0.1, 0.3]]])
        testing_utils.assert_close_tf(swap_xy(a), b)
