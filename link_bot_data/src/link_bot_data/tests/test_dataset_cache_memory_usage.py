#!/usr/bin/env python
import os
import pathlib
import unittest

import progressbar
import psutil

from link_bot_data.classifier_dataset import ClassifierDatasetLoader


def iterate_and_record_max_ram_usage(dataset):
    tf_dataset = dataset
    batch_size = 256
    tf_dataset = tf_dataset.batch(batch_size).take(100)

    max_ram_usage = 0
    for _ in progressbar.progressbar(tf_dataset):
        process = psutil.Process(os.getpid())
        current_ram_usage = process.memory_info().rss
        max_ram_usage = max(current_ram_usage, max_ram_usage)
    return max_ram_usage


class MyTestCase(unittest.TestCase):
    def test_mem_usage(self):
        no_cache_dataset = ClassifierDatasetLoader([pathlib.Path('classifier_data/rope_relaxed_mer')])
        no_cache_dataset.cache_negative = False
        max_ram_no_cache = iterate_and_record_max_ram_usage(no_cache_dataset)
        del no_cache_dataset

        cache_dataset = ClassifierDatasetLoader([pathlib.Path('classifier_data/rope_relaxed_mer')])
        cache_dataset.cache_negative = True
        max_ram_yes_cache = iterate_and_record_max_ram_usage(cache_dataset)

        self.assertGreater(max_ram_yes_cache, max_ram_no_cache * 1.3)
