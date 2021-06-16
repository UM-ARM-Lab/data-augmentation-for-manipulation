#!/usr/bin/env python3

import argparse
import pathlib

from link_bot_data.split_dataset import split_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=pathlib.Path, help="directory of tfrecord files")
    args = parser.parse_args()

    split_dataset(args.dataset_dir)


if __name__ == '__main__':
    main()
