#!/usr/bin/env python
import colorama
import argparse
import pathlib

import numpy as np
import rospy
import tensorflow as tf

from arc_utilities import ros_init
from link_bot_classifiers import train_test_recovery
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(6)


def train_main(args):
    train_test_recovery.train_main(**vars(args))


def eval_main(args):
    train_test_recovery.eval_main(**vars(args))


def eval_ensemble_main(args):
    train_test_recovery.eval_ensemble_main(**vars(args))


@ros_init.with_ros("train_test_recovery")
def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    train_parser.add_argument('model_hparams', type=pathlib.Path)
    train_parser.add_argument('classifier_checkpoint', type=pathlib.Path)
    train_parser.add_argument('--checkpoint', type=pathlib.Path)
    train_parser.add_argument('--batch-size', type=int, default=64)
    train_parser.add_argument('--take', type=int)
    train_parser.add_argument('--epochs', type=int, default=10)
    train_parser.add_argument('--no-validate', action='store_true')
    train_parser.add_argument('--log', '-l')
    train_parser.add_argument('--verbose', '-v', action='count', default=0)
    train_parser.add_argument('--log-scalars-every', type=int, help='loss every this many steps/batches', default=100)
    train_parser.add_argument('--validation-every', type=int, help='report validation every this many epochs',
                              default=1)
    train_parser.add_argument('--seed', type=int, default=None)
    train_parser.set_defaults(func=train_main)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    eval_parser.add_argument('checkpoint', type=pathlib.Path)
    eval_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val', 'all'], default='test')
    eval_parser.add_argument('--batch-size', type=int, default=64)
    eval_parser.add_argument('--verbose', '-v', action='count', default=0)
    eval_parser.add_argument('--seed', type=int, default=None)
    eval_parser.add_argument('--take', type=int)
    eval_parser.set_defaults(func=eval_main)

    eval_ensemble_parser = subparsers.add_parser('eval_ensemble')
    eval_ensemble_parser.add_argument('dataset_dir', type=pathlib.Path)
    eval_ensemble_parser.add_argument('ensemble_path', type=pathlib.Path)
    eval_ensemble_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val', 'all'], default='val')
    eval_ensemble_parser.add_argument('--batch-size', type=int, default=64)
    eval_ensemble_parser.add_argument('--verbose', '-v', action='count', default=0)
    eval_ensemble_parser.add_argument('--take', type=int)
    eval_ensemble_parser.add_argument('--use-gt-rope', action='store_true')
    eval_ensemble_parser.set_defaults(func=eval_ensemble_main)

    args = parser.parse_args()

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
