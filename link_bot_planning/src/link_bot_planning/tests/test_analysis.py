#!/usr/bin/env python

import unittest

import numpy as np

from link_bot_planning.analysis.figspec import make_groups


class TestCase(unittest.TestCase):

    def test_make_groups1(self):
        data = np.array([
            [0, 0, 0, 20],
            [0, 0, 1, 10],
            [0, 0, 2, 13],
            [0, 1, 0, 30],
            [0, 1, 1, 20],
            [0, 2, 0, 15],
        ])
        expected_groups = [
            np.array([
                [0, 0, 0, 20],
                [0, 0, 1, 10],
                [0, 0, 2, 13],
            ]),
            np.array([
                [0, 1, 0, 30],
                [0, 1, 1, 20],
            ]),
            np.array([
                [0, 2, 0, 15],
            ]),
        ]
        groups = make_groups(data, n_matching_dims=2)
        self.assertEqual(len(expected_groups), len(groups))
        for group_i, expected_group_i in zip(groups, expected_groups):
            np.testing.assert_equal(group_i, expected_group_i)

    def test_make_groups2(self):
        data = np.array([
            [3, 2, 0, 30],
            [3, 2, 1, 20],
        ])
        expected_groups = [
            np.array([
                [3, 2, 0, 30],
                [3, 2, 1, 20],
            ]),
        ]
        groups = make_groups(data, n_matching_dims=2)
        self.assertEqual(len(expected_groups), len(groups))
        for group_i, expected_group_i in zip(groups, expected_groups):
            np.testing.assert_equal(group_i, expected_group_i)


if __name__ == '__main__':
    unittest.main()
