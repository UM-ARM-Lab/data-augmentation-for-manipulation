#!/usr/bin/env python
import argparse
import logging
import pathlib
import time

import tensorflow as tf
from colorama import Fore
from mde.make_mde_dataset import make_mde_dataset

import rospy
from arc_utilities import ros_init
from arc_utilities.filesystem_utils import mkdir_and_ask


@ros_init.with_ros("make_mde_dataset")
def main():
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('checkpoint', type=str, help='dynamics model checkpoint')
    parser.add_argument('out_dir', type=pathlib.Path, help='out dir')
    parser.add_argument('--batch-size', type=int, help='batch size', default=8)
    parser.add_argument('--yes', '-y', action='store_true')

    args = parser.parse_args()

    outdir = args.out_dir.parent / (args.out_dir.name + f'_{int(time.time())}')
    success = mkdir_and_ask(outdir, parents=True, yes=args.yes)
    if not success:
        print(Fore.RED + "Aborting" + Fore.RESET)
        return

    rospy.loginfo(Fore.GREEN + f"Writing MERP dataset to {outdir}")
    make_mde_dataset(dataset_dir=args.dataset_dir,
                     checkpoint=args.checkpoint,
                     outdir=outdir)


if __name__ == '__main__':
    main()
