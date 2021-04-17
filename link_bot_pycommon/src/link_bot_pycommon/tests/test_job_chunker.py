import pathlib
import tempfile
import unittest

import hjson

from link_bot_pycommon.job_chunking import JobChunker


class Test(unittest.TestCase):
    def test_no_hierarchy(self):
        logfile_name = pathlib.Path(tempfile.gettempdir()) / 'test'
        logfile_name.unlink(missing_ok=True)

        c = JobChunker(logfile_name)
        for i in range(2):
            key = f'{i}'
            if c.has_result(key):
                continue
            c.store_result(key, i ** 2)

        with logfile_name.open('r') as logfile:
            log = hjson.load(logfile)
        self.assertIn('0', log)
        self.assertIn('1', log)
        self.assertNotIn('2', log)
        self.assertNotIn('3', log)

        c = JobChunker(logfile_name)
        for i in range(2, 4):
            key = f'{i}'
            if c.has_result(key):
                continue
            c.store_result(key, i ** 2)

        with logfile_name.open('r') as logfile:
            log = hjson.load(logfile)
        self.assertIn('0', log)
        self.assertIn('1', log)
        self.assertIn('2', log)
        self.assertIn('3', log)

        c = JobChunker(logfile_name)
        for i in range(4):
            key = f'{i}'
            self.assertTrue(c.has_result(key))

    def test_one_level_hierarchy(self):
        logfile_name = pathlib.Path(tempfile.gettempdir()) / 'test2'
        logfile_name.unlink(missing_ok=True)

        c1 = JobChunker(logfile_name)
        for i in range(2):
            k1 = f'{i}'
            c1.setup_key(k1)
            c2 = c1.sub_chunker(k1)

            for l in ['a', 'b', 'c']:
                k2 = l
                if c2.has_result(k2):
                    print(f"{k2=} exists, skipping")
                    continue

                c2.store_result(k2, {l: i ** 2})

        with logfile_name.open('r') as logfile:
            log = hjson.load(logfile)
        self.assertEqual(log['0']['a'], {'a': 0})
        self.assertEqual(log['0']['b'], {'b': 0})
        self.assertEqual(log['1']['a'], {'a': 1})


if __name__ == '__main__':
    unittest.main()
