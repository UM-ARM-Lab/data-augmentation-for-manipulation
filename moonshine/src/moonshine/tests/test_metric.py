#!/usr/bin/env python
import unittest

import tensorflow as tf

from moonshine.metric import fn_rate, fp_rate


class Test(unittest.TestCase):

    def test_false_negative_rate(self):
        fnr = fn_rate(y_pred=tf.constant([0.6, 0.8, 0.2, 0.4, 0.9, 0.3]), y_true=tf.constant([1.0, 0, 1, 0, 0, 1]))
        self.assertAlmostEqual(fnr.numpy(), 2.0 / 4.0)

    def test_false_positive_rate(self):
        fpr = fp_rate(y_pred=tf.constant([0.6, 0.8, 0.2, 0.4, 0.9, 0.3]), y_true=tf.constant([1.0, 0, 1, 0, 0, 1]))
        self.assertAlmostEqual(fpr.numpy(), 2.0 / 4.0)

    def test_rates(self):
        y_true = tf.constant([1.0, 0, 1, 0, 0, 1, 0, 1, 0, 1])

        for i in range(10):
            y_pred = tf.random.uniform([10], 0, 1)
            fnr = fn_rate(y_pred, y_true)
            fpr = fp_rate(y_pred, y_true)
            self.assertAlmostEqual(fnr.numpy() + fpr.numpy(), 1.0)
