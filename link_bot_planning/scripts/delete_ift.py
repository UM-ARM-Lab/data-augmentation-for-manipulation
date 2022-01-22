#!/usr/bin/env python
import argparse
import pathlib

from arc_utilities.path_utils import rm_tree
from link_bot_pycommon.args import int_set_arg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=pathlib.Path)
    parser.add_argument('iters', type=int_set_arg)

    args = parser.parse_args()

    for i in args.iters:
        planning_dir = args.dir / "planning_results" / f"iteration_{i:04d}_planning"
        training_dir = args.dir / "training_logdir" / f"iteration_{i:04d}_classifier_training_logdir"
        dataset_dir = args.dir / "classifier_dataset" / f"iteration_{i:04d}_dataset"
        dataset_aug_dir = args.dir / "classifier_datasets_aug" / f"iteration_{i:04d}_dataset"

        dirs_to_remove = [
            planning_dir,
            training_dir,
            dataset_dir,
            dataset_aug_dir,
        ]

        for d in dirs_to_remove:
            if d.exists():
                try:
                    rm_tree(d)
                    print(f"Removed {d}")
                except FileNotFoundError:
                    pass
            else:
                print(f"{d} does not exist")


if __name__ == '__main__':
    main()
