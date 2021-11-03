#!/usr/bin/env python
import argparse
import pathlib
from time import time

import numpy as np

from arc_utilities import ros_init
from link_bot_pycommon.args import run_subparsers
from propnet import train_test_propnet


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
    train_parser.add_argument('dataset_dir', type=pathlib.Path)
    train_parser.add_argument('model_params', type=pathlib.Path)  # hparams/blocks.hjson
    train_parser.add_argument('--checkpoint', type=pathlib.Path)
    train_parser.add_argument('--batch-size', type=int, default=24)
    train_parser.add_argument('--take', type=int)
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--log', '-l')
    train_parser.add_argument('--no-validate', action='store_true')
    train_parser.add_argument('--seed', type=int, default=None)
    train_parser.set_defaults(func=train_main)

    viz_parser = subparsers.add_parser('viz')
    viz_parser.add_argument('dataset_dir', type=pathlib.Path)
    viz_parser.add_argument('checkpoint', type=pathlib.Path)
    viz_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val', 'all'], default='val')
    viz_parser.set_defaults(func=viz_main)

    run_subparsers(parser)


if __name__ == '__main__':
    main()
