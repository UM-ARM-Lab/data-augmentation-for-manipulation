#!/usr/bin/env python
import argparse
import logging
import pathlib

import colorama
import tensorflow as tf
from colorama import Fore

from arc_utilities import ros_init
from arc_utilities.filesystem_utils import mkdir_and_ask
from link_bot_data.recovery_dataset_utils import make_recovery_dataset
from link_bot_pycommon.args import my_formatter


@ros_init.with_ros("make_recovery_dataset")
def main():
    colorama.init(autoreset=True)

    tf.get_logger().setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('labeling_params', type=pathlib.Path)
    parser.add_argument('fwd_model_dir', type=pathlib.Path, help='forward model', nargs="+")
    parser.add_argument('classifier_model_dir', type=pathlib.Path)
    parser.add_argument('out_dir', type=pathlib.Path, help='out dir')
    parser.add_argument('--start-at', type=str, help='string of the form mode,idx (ex: val:30)')
    parser.add_argument('--stop-at', type=str, help='string of the form mode,idx (ex: val:30)')
    parser.add_argument('--batch-size', type=int, help="batch size", default=2)
    parser.add_argument('--yes', action='store_true')

    args = parser.parse_args()

    success = mkdir_and_ask(args.out_dir, parents=True, yes=args.yes)
    if not success:
        print(Fore.RED + "Aborting" + Fore.RESET)
        return

    make_recovery_dataset(dataset_dir=args.dataset_dir,
                          fwd_model_dir=args.fwd_model_dir,
                          classifier_model_dir=args.classifier_model_dir,
                          use_gt_rope=True,
                          labeling_params=args.labeling_params,
                          outdir=args.out_dir,
                          batch_size=args.batch_size,
                          start_at=args.start_at,
                          stop_at=args.stop_at)


if __name__ == '__main__':
    main()
