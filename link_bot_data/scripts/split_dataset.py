#!/usr/bin/env python3

import argparse
import pathlib

from link_bot_data.dataset_utils import DEFAULT_VAL_SPLIT, DEFAULT_TEST_SPLIT
from link_bot_data.files_dataset import FilesDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=pathlib.Path, help="directory of tfrecord files")
    parser.add_argument("--extension", type=pathlib.Path, help="extension of the data files", default='tfrecords')
    parser.add_argument("--fraction-validation", '-v', type=float, help="fraction of files to put in validation",
                        default=DEFAULT_TEST_SPLIT)
    parser.add_argument("--fraction-testing", '-t', type=float, help="fraction of files to put in validation",
                        default=DEFAULT_VAL_SPLIT)
    args = parser.parse_args()

    files_dataset = FilesDataset(root_dir=args.dataset_dir)
    sorted_records = sorted(list(args.dataset_dir.glob(f"example_*.{args.extension}")))
    for file in sorted_records:
        files_dataset.add(file)
    files_dataset.split()


if __name__ == '__main__':
    main()
