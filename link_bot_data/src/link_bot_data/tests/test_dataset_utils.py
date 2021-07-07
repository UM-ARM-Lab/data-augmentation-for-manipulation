import unittest
from io import BytesIO

import numpy as np
import tensorflow as tf

from link_bot_data.dataset_utils import is_reconverging, null_pad, NULL_PAD_VALUE, num_reconverging, \
    num_reconverging_subsequences, add_predicted, remove_predicted, replaced_true_with_predicted, multigen, \
    compute_batch_size_for_n_examples, deserialize_scene_msg
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import remove_batch
from moveit_msgs.msg import PlanningScene

limit_gpu_mem(0.1)


@multigen
def make_dataset():
    for i in range(10):
        yield i


class MyTestCase(unittest.TestCase):
    def test_is_reconverging(self):
        batch_is_reconverging_output = is_reconverging(
            tf.constant([[1, 0, 0, 1], [1, 1, 1, 0], [1, 0, 0, 0]], tf.int64)).numpy()
        self.assertTrue(batch_is_reconverging_output[0])
        self.assertFalse(batch_is_reconverging_output[1])
        self.assertFalse(batch_is_reconverging_output[2])
        self.assertTrue(remove_batch(is_reconverging(tf.constant([[1, 0, 0, 1]], tf.int64))).numpy())
        self.assertFalse(remove_batch(is_reconverging(tf.constant([[1, 0, 0, 0]], tf.int64))).numpy())
        self.assertFalse(remove_batch(is_reconverging(tf.constant([[1, 0, 1, 0]], tf.int64))).numpy())

    def test_num_reconverging_subsequences(self):
        self.assertEqual(
            num_reconverging_subsequences(tf.constant([[1, 0, 0, 1], [1, 1, 1, 0], [1, 0, 1, 1]], tf.int64)).numpy(),
            3)
        self.assertEqual(num_reconverging_subsequences(tf.constant([[1, 1, 0, 1, 1, 1]], tf.int64)).numpy(), 6)
        self.assertEqual(num_reconverging_subsequences(tf.constant([[1, 0, 0, 0]], tf.int64)).numpy(), 0)

    def test_num_reconverging(self):
        self.assertEqual(num_reconverging(tf.constant([[1, 0, 0, 1], [1, 1, 1, 0], [1, 0, 1, 1]], tf.int64)).numpy(), 2)
        self.assertEqual(num_reconverging(tf.constant([[1, 0, 0, 1]], tf.int64)).numpy(), 1)
        self.assertEqual(num_reconverging(tf.constant([[1, 0, 0, 0]], tf.int64)).numpy(), 0)

    def test_null_pad(self):
        np.testing.assert_allclose(null_pad(np.array([1, 0, 0, 1]), start=0, end=2),
                                   np.array([1, 0, 0, NULL_PAD_VALUE]))
        np.testing.assert_allclose(null_pad(np.array([1, 0, 0, 1]), start=0, end=3),
                                   np.array([1, 0, 0, 1]))
        np.testing.assert_allclose(null_pad(np.array([1, 0, 0, 1]), start=1, end=2),
                                   np.array([NULL_PAD_VALUE, 0, 0, NULL_PAD_VALUE]))
        np.testing.assert_allclose(null_pad(np.array([1, 0, 0, 1]), start=1, end=3),
                                   np.array([NULL_PAD_VALUE, 0, 0, 1]))
        np.testing.assert_allclose(null_pad(np.array([1, 0, 0, 1]), start=2, end=3),
                                   np.array([NULL_PAD_VALUE, NULL_PAD_VALUE, 0, 1]))
        np.testing.assert_allclose(null_pad(np.array([1, 0, 0, 1]), start=2, end=2),
                                   np.array([NULL_PAD_VALUE, NULL_PAD_VALUE, 0, NULL_PAD_VALUE]))
        np.testing.assert_allclose(null_pad(np.array([1, 0, 0, 1]), start=2, end=None),
                                   np.array([NULL_PAD_VALUE, NULL_PAD_VALUE, 0, 1]))
        np.testing.assert_allclose(null_pad(np.array([1, 0, 0, 1]), start=None, end=None),
                                   np.array([1, 0, 0, 1]))
        np.testing.assert_allclose(null_pad(np.array([1, 0, 0, 1]), start=None, end=0),
                                   np.array([1, NULL_PAD_VALUE, NULL_PAD_VALUE, NULL_PAD_VALUE]))

    def test_add_remove_predicted(self):
        k = "test"
        out_k = remove_predicted(add_predicted(k))
        self.assertEqual(k, out_k)

    def test_add_remove_predicted_dict(self):
        d = {
            add_predicted("test1"): 1,
            "test1": 3,
            "test2":                2,
        }
        expected_d = {
            "test1": 1,
            "test2": 2,
        }
        out_d = replaced_true_with_predicted(d)
        self.assertEqual(expected_d, out_d)

    def test_deserialize_scene_msg(self):
        d = {'scene_msg': PlanningScene()}
        deserialize_scene_msg(d)
        self.assertIsInstance(d['scene_msg'], PlanningScene)

        d = {'scene_msg': [PlanningScene()]}
        deserialize_scene_msg(d)
        self.assertIsInstance(d['scene_msg'], list)
        self.assertIsInstance(d['scene_msg'][0], PlanningScene)

        d = {'scene_msg': np.array([PlanningScene()])}
        deserialize_scene_msg(d)
        self.assertIsInstance(d['scene_msg'], np.ndarray)
        self.assertIsInstance(d['scene_msg'][0], PlanningScene)

        msg = PlanningScene()
        buff = BytesIO()
        msg.serialize(buff)
        serialized_bytes = buff.getvalue()
        z = tf.convert_to_tensor(serialized_bytes)
        d = {'scene_msg': z}
        deserialize_scene_msg(d)
        self.assertIsInstance(d['scene_msg'], PlanningScene)

        d = {'scene_msg': z[tf.newaxis]}
        deserialize_scene_msg(d)
        self.assertIsInstance(d['scene_msg'], np.ndarray)
        self.assertIsInstance(d['scene_msg'][0], PlanningScene)

    def test_multigen(self):
        dataset = make_dataset()

        # iterate once
        for i in dataset:
            pass

        # test that iterating again works
        second_time_iter = iter(dataset)
        self.assertEqual(next(second_time_iter), 0)

    def test_compute_batch_size(self):
        self.assertEqual(compute_batch_size_for_n_examples(1, 16), 1)
        self.assertEqual(compute_batch_size_for_n_examples(2, 16), 1)
        self.assertEqual(compute_batch_size_for_n_examples(4, 16), 2)
        self.assertEqual(compute_batch_size_for_n_examples(8, 16), 4)
        self.assertEqual(compute_batch_size_for_n_examples(16, 16), 8)
        self.assertEqual(compute_batch_size_for_n_examples(32, 16), 16)


if __name__ == '__main__':
    unittest.main()
