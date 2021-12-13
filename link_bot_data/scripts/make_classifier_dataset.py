#!/usr/bin/env python
import argparse
import logging
import pathlib
import time

import tensorflow as tf
from colorama import Fore

import rospy
from arc_utilities import ros_init
from arc_utilities.filesystem_utils import mkdir_and_ask
from link_bot_data.classifier_dataset_utils import make_classifier_dataset


@ros_init.with_ros("make_classifier_dataset")
def main():
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('labeling_params', type=pathlib.Path)
    parser.add_argument('fwd_model_dir', type=pathlib.Path, help='forward model')
    parser.add_argument('out_dir', type=pathlib.Path, help='out dir')
    parser.add_argument('--total-take', type=int, help="will be split up between train/test/val")
    parser.add_argument('--start-at', type=str, help='mode:batch_index, ex train:10')
    parser.add_argument('--stop-at', type=str, help='mode:batch_index, ex train:10')
    parser.add_argument('--batch-size', type=int, help='batch size', default=8)
    parser.add_argument('--threshold', type=float, help='threshold')
    parser.add_argument('--yes', '-y', action='store_true')
    parser.add_argument('--use-gt-rope', action='store_true')
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    outdir = args.out_dir.parent / (args.out_dir.name + f'_{int(time.time())}')
    success = mkdir_and_ask(outdir, parents=True, yes=args.yes)
    if not success:
        print(Fore.RED + "Aborting" + Fore.RESET)
        return

    rospy.loginfo(Fore.GREEN + f"Writing classifier dataset to {outdir}")
    make_classifier_dataset(dataset_dir=args.dataset_dir,
                            fwd_model_dir=args.fwd_model_dir,
                            labeling_params=args.labeling_params,
                            outdir=outdir,
                            use_gt_rope=args.use_gt_rope,
                            visualize=args.visualize,
                            save_format='pkl',
                            batch_size=args.batch_size,
                            start_at=args.start_at,
                            custom_threshold=args.threshold,
                            stop_at=args.stop_at)


if __name__ == '__main__':
    main()
