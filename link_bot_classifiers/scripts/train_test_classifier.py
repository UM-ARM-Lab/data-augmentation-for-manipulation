#!/usr/bin/env python
import argparse
import logging
import pathlib
from time import time

import numpy as np
import tensorflow as tf

from arc_utilities import ros_init
from link_bot_classifiers import train_test_classifier
from link_bot_classifiers.train_test_classifier import add_eval_args
from link_bot_pycommon.args import run_subparsers, int_tuple_arg
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


def train_main(args):
    if args.seed is None:
        args.seed = np.random.randint(0, 10000)

    print("Using seed {}".format(args.seed))
    train_test_classifier.train_main(**vars(args))


def train_n_main(args):
    for i in range(args.n):
        args.log = f"{args.log}{i}"
        args.seed = i
        train_test_classifier.train_main(**vars(args))


def compare_main(args):
    train_test_classifier.compare_main(**vars(args))


def eval_main(args):
    train_test_classifier.eval_main(**vars(args))


def eval_n_main(args):
    train_test_classifier.eval_n_main(**vars(args))


def viz_main(args):
    train_test_classifier.viz_main(**vars(args))


def eval_ensemble_main(args):
    train_test_classifier.eval_ensemble_main(**vars(args))


def viz_ensemble_main(args):
    train_test_classifier.viz_ensemble_main(**vars(args))


node_name = f"train_test_classifier_{int(time())}"


@ros_init.with_ros(node_name)
def main():
    tf.get_logger().setLevel(logging.ERROR)
    np.set_printoptions(linewidth=250, precision=4, suppress=True)
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    train_parser.add_argument('model_hparams', type=pathlib.Path)
    train_parser.add_argument('--checkpoint', type=pathlib.Path)
    train_parser.add_argument('--batch-size', type=int, default=24)
    train_parser.add_argument('--take', type=int)
    train_parser.add_argument('--debug', action='store_true')
    train_parser.add_argument('--epochs', type=int, default=10)
    train_parser.add_argument('--log', '-l')
    train_parser.add_argument('--verbose', '-v', action='count', default=0)
    train_parser.add_argument('--no-validate', action='store_true')
    train_parser.add_argument('--save-inputs', action='store_true')
    train_parser.add_argument('--ensemble-idx', type=int)
    train_parser.add_argument('--log-scalars-every', type=int,
                              help='loss/accuracy every this many steps/batches', default=100)
    train_parser.add_argument('--validation-every', type=int,
                              help='report validation every this many epochs', default=1)
    train_parser.add_argument('--seed', type=int, default=None)
    train_parser.add_argument('--threshold', type=float, default=None)
    train_parser.add_argument('--augmentation-config-dir', type=pathlib.Path, help='dir of pkl files with state/env')
    train_parser.set_defaults(func=train_main)

    train_n_parser = subparsers.add_parser('train_n')
    train_n_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    train_n_parser.add_argument('model_hparams', type=pathlib.Path)
    train_n_parser.add_argument('n', type=int, help='number of models to train')
    train_n_parser.add_argument('log', help='used in naming the model')
    train_n_parser.add_argument('--batch-size', type=int, default=24)
    train_n_parser.add_argument('--epochs', type=int, default=10)
    train_n_parser.add_argument('--verbose', '-v', action='count', default=0)
    train_n_parser.set_defaults(func=train_n_main)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    eval_parser.add_argument('checkpoint', type=pathlib.Path)
    eval_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val', 'all'], default='all')
    eval_parser.add_argument('--balance', action='store_true')
    eval_parser.add_argument('--batch-size', type=int, default=128)
    eval_parser.add_argument('--verbose', '-v', action='count', default=0)
    eval_parser.add_argument('--take', type=int)
    eval_parser.add_argument('--threshold', type=float, default=None)
    eval_parser.add_argument('--debug', action='store_true')
    eval_parser.add_argument('--profile', type=int_tuple_arg, default=None)
    eval_parser.set_defaults(func=eval_main)

    eval_n_parser = subparsers.add_parser('eval_n')
    eval_n_parser.add_argument('--dataset-dirs', type=pathlib.Path, nargs='+', required=True)
    eval_n_parser.add_argument('--checkpoints', type=pathlib.Path, nargs='+', required=True)
    eval_n_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val', 'all'], default='all')
    add_eval_args(eval_n_parser)
    eval_n_parser.set_defaults(func=eval_n_main)

    compare_parser = subparsers.add_parser('compare')
    compare_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    compare_parser.add_argument('checkpoint1', type=pathlib.Path)
    compare_parser.add_argument('checkpoint2', type=pathlib.Path)
    compare_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val', 'all'], default='val')
    compare_parser.add_argument('--batch-size', type=int, default=32)
    compare_parser.add_argument('--verbose', '-v', action='count', default=0)
    compare_parser.add_argument('--take', type=int)
    compare_parser.add_argument('--threshold', type=float, default=None)
    compare_parser.set_defaults(func=compare_main)

    viz_parser = subparsers.add_parser('viz')
    viz_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    viz_parser.add_argument('checkpoint', type=pathlib.Path)
    viz_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val', 'all'], default='val')
    viz_parser.add_argument('--batch-size', type=int, default=32)
    viz_parser.add_argument('--verbose', '-v', action='count', default=0)
    viz_parser.add_argument('--only-negative', action='store_true')
    viz_parser.add_argument('--only-positive', action='store_true')
    viz_parser.add_argument('--only-errors', action='store_true')
    viz_parser.add_argument('--only-fp', action='store_true')
    viz_parser.add_argument('--only-fn', action='store_true')
    viz_parser.add_argument('--only-tn', action='store_true')
    viz_parser.add_argument('--only-tp', action='store_true')
    viz_parser.add_argument('--threshold', type=float, default=None)
    viz_parser.add_argument('--start-at', type=int, default=0)
    viz_parser.set_defaults(func=viz_main)

    eval_ensemble_parser = subparsers.add_parser('eval_ensemble')
    eval_ensemble_parser.add_argument('dataset_dir', type=pathlib.Path)
    eval_ensemble_parser.add_argument('ensemble_path', type=pathlib.Path)
    eval_ensemble_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val', 'all'], default='val')
    eval_ensemble_parser.add_argument('--batch-size', type=int, default=64)
    eval_ensemble_parser.add_argument('--verbose', '-v', action='count', default=0)
    eval_ensemble_parser.add_argument('--take', type=int)
    eval_ensemble_parser.set_defaults(func=eval_ensemble_main)

    viz_ensemble_parser = subparsers.add_parser('viz_ensemble')
    viz_ensemble_parser.add_argument('dataset_dir', type=pathlib.Path)
    viz_ensemble_parser.add_argument('ensemble_path', type=pathlib.Path)
    viz_ensemble_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val', 'all'], default='val')
    viz_ensemble_parser.add_argument('--batch-size', type=int, default=64)
    viz_ensemble_parser.add_argument('--verbose', '-v', action='count', default=0)
    viz_ensemble_parser.add_argument('--take', type=int)
    viz_ensemble_parser.add_argument('--stdev', type=str)
    viz_ensemble_parser.set_defaults(func=viz_ensemble_main)

    run_subparsers(parser)


if __name__ == '__main__':
    main()
