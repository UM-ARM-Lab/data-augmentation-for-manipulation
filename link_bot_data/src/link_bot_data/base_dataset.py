#!/usr/bin/env python
from __future__ import annotations

import csv
import pathlib
from typing import List, Optional, Dict, Callable, Any

import numpy as np
import tensorflow as tf

from link_bot_data.dataset_utils import parse_and_deserialize, make_add_batch_func, parse_and_slow_deserialize, \
    multigen, merge_hparams_dicts, label_is

SORT_FILE_NAME = 'sort_order.csv'


class SizedTFDataset:

    def __init__(self, dataset: tf.data.Dataset, records: List, size: Optional[int] = None):
        self.dataset = dataset
        self.records = records
        if size is None:  # do I really want to do this?
            self.size = len(self.records)
        else:
            self.size = size

    def __len__(self):
        return self.size

    def __iter__(self):
        return self.dataset.__iter__()

    def __repr__(self):
        if len(self.records) > 0:
            dir = pathlib.Path(self.records[0]).parent.as_posix()
        else:
            dir = "??"
        return f"Dataset: {dir}, size={self.size}"

    def batch(self, batch_size: int, *args, **kwargs):
        dataset_batched = self.dataset.batch(*args, batch_size=batch_size, **kwargs)
        dataset_batched = dataset_batched.map(make_add_batch_func(batch_size))
        return SizedTFDataset(dataset_batched, self.records, size=int(self.size / batch_size))

    def mymap(self, function: Callable, *args, **kwargs):
        @multigen
        def _mymap():
            for e in self.dataset:
                yield function(e, *args, **kwargs)

        mapped_generator = _mymap()
        return SizedTFDataset(mapped_generator, self.records)

    def map(self, function: Callable):
        dataset_mapped = self.dataset.map(function)
        return SizedTFDataset(dataset_mapped, self.records)

    def filter(self, function: Callable):
        dataset_filter = self.dataset.filter(function)
        return SizedTFDataset(dataset_filter, self.records, size=None)

    def repeat(self, count: Optional[int] = None):
        dataset = self.dataset.repeat(count)
        return SizedTFDataset(dataset, self.records, size=None)

    def shuffle(self, buffer_size: int, seed: Optional[int] = None, reshuffle_each_iteration: bool = False):
        dataset = self.dataset.shuffle(buffer_size, seed, reshuffle_each_iteration)
        return SizedTFDataset(dataset, self.records)

    def prefetch(self, buffer_size: Any = tf.data.experimental.AUTOTUNE):
        dataset = self.dataset.prefetch(buffer_size)
        return SizedTFDataset(dataset, self.records)

    def take(self, count: int):
        if count is not None:
            dataset = self.dataset.take(count)
            return SizedTFDataset(dataset, self.records, size=count)
        return self

    def zip(self, dataset2: SizedTFDataset):
        dataset = tf.data.Dataset.zip((self.dataset, dataset2.dataset))
        return SizedTFDataset(dataset, self.records + dataset2.records, size=min(self.size, dataset2.size))

    def balance(self):
        # double it, because want to up-sample the underrepresented class. This approximates that.
        # If the under-represented class makes up >33% then this will be exact, if it's less the we will
        # start dropping examples from the over-represented class to make up for it.
        new_dataset_size = self.size * 2
        positive_dataset = self.dataset.filter(label_is(1))
        negative_dataset = self.dataset.filter(label_is(0))
        negative_dataset = negative_dataset.repeat()
        positive_dataset = positive_dataset.repeat()

        datasets = [positive_dataset, negative_dataset]
        balanced_dataset = tf.data.experimental.sample_from_datasets(datasets=datasets, weights=[0.5, 0.5])
        balanced_dataset = balanced_dataset.take(new_dataset_size)
        return SizedTFDataset(balanced_dataset, records=[], size=new_dataset_size)


class BaseDatasetLoader:

    def __init__(self, dataset_dirs: List[pathlib.Path], verbose: int = 0):
        self.verbose = verbose
        self.name = '-'.join([d.name for d in dataset_dirs])
        self.dataset_dirs = dataset_dirs
        self.hparams = merge_hparams_dicts(dataset_dirs, self.verbose)

        self.scenario_metadata = dict(self.hparams.get('scenario_metadata', {}))

    def get_datasets(self,
                     mode: str,
                     n_parallel_calls: int = tf.data.experimental.AUTOTUNE,
                     do_not_process: bool = False,
                     shard: Optional[int] = None,
                     take: Optional[int] = None,
                     shuffle: Optional[bool] = False,
                     sort: Optional[bool] = False,
                     **kwargs):
        all_filenames = self.get_record_filenames(mode, sort=sort)
        return self.get_datasets_from_records(all_filenames,
                                              n_parallel_calls=n_parallel_calls,
                                              do_not_process=do_not_process,
                                              shard=shard,
                                              shuffle=shuffle,
                                              take=take,
                                              **kwargs)

    def get_record_filenames(self, mode: str, sort: Optional[bool] = False):
        if mode == 'all':
            train_filenames = []
            test_filenames = []
            val_filenames = []
            for dataset_dir in self.dataset_dirs:
                train_filenames.extend(str(filename) for filename in dataset_dir.glob("{}/*.tfrecords".format('train')))
                test_filenames.extend(str(filename) for filename in dataset_dir.glob("{}/*.tfrecords".format('test')))
                val_filenames.extend(str(filename) for filename in dataset_dir.glob("{}/*.tfrecords".format('val')))

            all_filenames = train_filenames
            all_filenames.extend(test_filenames)
            all_filenames.extend(val_filenames)
        else:
            all_filenames = []
            for dataset_dir in self.dataset_dirs:
                all_filenames.extend(str(filename) for filename in (dataset_dir / mode).glob("*.tfrecords"))

        # Initially sort lexicographically, which is intended to make things read in the order they were written
        all_filenames = sorted(all_filenames)

        # Sorting
        if len(self.dataset_dirs) > 1 and sort:
            raise NotImplementedError("Don't know how to make a sorted multi-dataset")

        if sort:
            sorted_filename = self.dataset_dirs[0] / SORT_FILE_NAME
            with open(sorted_filename, "r") as sorted_file:
                reader = csv.reader(sorted_file)
                sort_info_filenames = [row[0] for row in reader]

                # sort all_filenames based on the order of sort_info
                def _sort_key(filename):
                    return sort_info_filenames.index(filename)

                all_filenames = sorted(all_filenames, key=_sort_key)

        return all_filenames

    def get_datasets_from_records(self,
                                  records: List[str],
                                  n_parallel_calls: Optional[int] = None,
                                  do_not_process: Optional[bool] = False,
                                  shard: Optional[int] = None,
                                  take: Optional[int] = None,
                                  shuffle: Optional[bool] = False,
                                  **kwargs,
                                  ):
        if shuffle:
            shuffle_rng = np.random.RandomState(0)
            shuffle_rng.shuffle(records)

        dataset = tf.data.TFRecordDataset(records, buffer_size=1 * 1024 * 1024, compression_type='ZLIB')

        # Given the member lists of states, actions, and constants set in the constructor, create a dict for parsing a feature
        features_description = self.make_features_description()
        if kwargs.get("slow", False):
            print("Using slow parsing!")
            dataset = parse_and_slow_deserialize(dataset, features_description, n_parallel_calls)
        else:
            dataset = parse_and_deserialize(dataset, features_description, n_parallel_calls)

        if take is not None:
            dataset = dataset.take(take)

        if shard is not None:
            dataset = dataset.shard(shard)

        if not do_not_process:
            dataset = self.post_process(dataset, n_parallel_calls)

        sized_tf_dataset = SizedTFDataset(dataset, records)
        return sized_tf_dataset

    def make_features_description(self):
        if 'has_tfrecord_path' in self.hparams and self.hparams['has_tfrecord_path']:
            return {'tfrecord_path': tf.io.FixedLenFeature([], tf.string)}
        else:
            return {}

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):
        scenario_metadata = self.scenario_metadata

        def _add_scenario_metadata(example: Dict):
            example.update(scenario_metadata)
            return example

        dataset = dataset.map(_add_scenario_metadata)
        return dataset
