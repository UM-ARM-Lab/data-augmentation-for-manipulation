#!/usr/bin/env python3

import argparse
import pathlib

from link_bot_data.dataset_utils import DEFAULT_VAL_SPLIT, DEFAULT_TEST_SPLIT
from link_bot_data.split_dataset import split_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=pathlib.Path, help="directory of tfrecord files")
    parser.add_argument("--test-split", type=float, default=DEFAULT_TEST_SPLIT)
    parser.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT)
    args = parser.parse_args()

    split_dataset(args.dataset_dir, val_split=args.val_split, test_split=args.test_split)


if __name__ == '__main__':
    main()
