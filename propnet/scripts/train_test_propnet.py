#!/usr/bin/env python
import argparse
import pathlib
from time import time

import numpy as np

from arc_utilities import ros_init
from propnet import train_test_propnet
from link_bot_pycommon.args import run_subparsers, int_tuple_arg


def train_main(args):
    if args.seed is None:
        args.seed = np.random.randint(0, 10000)

    print("Using seed {}".format(args.seed))
    train_test_propnet.train_main(**vars(args))


def eval_main(args):
    train_test_propnet.eval_main(**vars(args))


def viz_main(args):
    train_test_propnet.viz_main(**vars(args))


node_name = f"train_test_propnet_{int(time())}"


@ros_init.with_ros(node_name)
def main():
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
    train_parser.set_defaults(func=train_main)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    eval_parser.add_argument('checkpoint', type=pathlib.Path)
    eval_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val', 'all'], default='all')
    eval_parser.add_argument('--balance', action='store_true')
    eval_parser.add_argument('--batch-size', type=int, default=128)
    eval_parser.add_argument('--verbose', '-v', action='count', default=0)
    eval_parser.add_argument('--take', type=int)
    eval_parser.add_argument('--debug', action='store_true')
    eval_parser.add_argument('--profile', type=int_tuple_arg, default=None)
    eval_parser.set_defaults(func=eval_main)

    viz_parser = subparsers.add_parser('viz')
    viz_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    viz_parser.add_argument('checkpoint', type=pathlib.Path)
    viz_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val', 'all'], default='val')
    viz_parser.add_argument('--batch-size', type=int, default=32)
    viz_parser.add_argument('--verbose', '-v', action='count', default=0)
    viz_parser.add_argument('--start-at', type=int, default=0)
    viz_parser.set_defaults(func=viz_main)

    run_subparsers(parser)


if __name__ == '__main__':
    main()
