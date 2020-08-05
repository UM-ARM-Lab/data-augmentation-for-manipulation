#!/usr/bin/env python3

import argparse
import pathlib

from link_bot_data.base_dataset import DEFAULT_TEST_SPLIT, DEFAULT_VAL_SPLIT
from link_bot_data.files_dataset import FilesDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=pathlib.Path, help="directory of tfrecord files")
    parser.add_argument("--fraction-validation", '-v', type=float, help="fraction of files to put in validation",
                        default=DEFAULT_TEST_SPLIT)
    parser.add_argument("--fraction-testing", '-t', type=float, help="fraction of files to put in validation",
                        default=DEFAULT_VAL_SPLIT)
    args = parser.parse_args()

    files_dataset = FilesDataset(root_dir=args.dataset_dir)
    files_dataset.split()


if __name__ == '__main__':
    main()
