#!/usr/bin/env python

import unittest

from link_bot_planning.my_planner import LoggingTree


class TestLoggingTree(unittest.TestCase):

    def test_logging_tree(self):
        t = LoggingTree()
        t.add(1, "1->2", 2)
        self.assertEqual(str(t), "21")
        t.add(2, "2->3", 3)
        self.assertEqual(str(t), "321")
        t.add(1, "1->4", 4)
        self.assertEqual(str(t), "3241")
        t.add(4, "4->5", 5)
        self.assertEqual(str(t), "32541")
        t.add(2, "2->6", 6)
        self.assertEqual(str(t), "362541")


if __name__ == '__main__':
    unittest.main()
