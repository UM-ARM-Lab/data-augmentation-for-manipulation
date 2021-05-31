#!/usr/bin/env python3

import argparse
import pathlib

from link_bot_data.dataset_utils import DEFAULT_VAL_SPLIT, DEFAULT_TEST_SPLIT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=pathlib.Path, help="directory of tfrecord files")
    parser.add_argument("--extension", type=pathlib.Path, help="extension of the data files", default='hjson')
    parser.add_argument("--fraction-validation", '-v', type=float, help="fraction of files to put in validation",
                        default=DEFAULT_TEST_SPLIT)
    parser.add_argument("--fraction-testing", '-t', type=float, help="fraction of files to put in validation",
                        default=DEFAULT_VAL_SPLIT)
    args = parser.parse_args()

    paths = sorted(list(args.dataset_dir.glob(f"example_*.{args.extension}")))

    n_files = len(paths)
    n_validation = int(DEFAULT_VAL_SPLIT * n_files)
    n_testing = int(DEFAULT_TEST_SPLIT * n_files)

    val_files = paths[0:n_validation]
    paths = paths[n_validation:]

    test_files = paths[0:n_testing]
    train_files = paths[n_testing:]

    def _write_mode(_filenames, mode):
        with (args.dataset_dir / f"{mode}.txt").open("w") as f:
            for _filename in _filenames:
                f.write(_filename.as_posix() + '\n')

    _write_mode(train_files, 'train')
    _write_mode(test_files, 'test')
    _write_mode(val_files, 'val')


if __name__ == '__main__':
    main()
