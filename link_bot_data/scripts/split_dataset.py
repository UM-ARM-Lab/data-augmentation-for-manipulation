#!/usr/bin/env python3

import argparse
import os
import pathlib
import random
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=pathlib.Path, help="directory of tfrecord files")
    parser.add_argument("--fraction-validation", '-v', type=float, help="fraction of files to put in validation", default=0.05)
    parser.add_argument("--fraction-testing", '-t', type=float, help="fraction of files to put in validation", default=0.05)
    args = parser.parse_args()

    os.makedirs(args.dataset_dir / 'train', exist_ok=True)
    os.makedirs(args.dataset_dir / 'test', exist_ok=True)
    os.makedirs(args.dataset_dir / 'val', exist_ok=True)

    all_files = list(args.dataset_dir.glob("*.tfrecords"))
    n_files = len(all_files)
    n_validation = int(args.fraction_validation * n_files)
    n_testing = int(args.fraction_testing * n_files)
    random.shuffle(all_files)
    val_files = all_files[0:n_validation]
    all_files = all_files[n_validation:]
    random.shuffle(all_files)
    test_files = all_files[0:n_testing]
    train_files = all_files[n_testing:]

    for train_file in train_files:
        out = args.dataset_dir / 'train' / train_file.name
        shutil.move(train_file, out)
    for test_file in test_files:
        out = args.dataset_dir / 'test' / test_file.name
        shutil.move(test_file, out)
    for val_file in val_files:
        out = args.dataset_dir / 'val' / val_file.name
        shutil.move(val_file, out)


if __name__ == '__main__':
    main()
