#!/usr/bin/env python
import argparse
import pathlib

import numpy as np
import tensorflow as tf

from arc_utilities import ros_init
from learn_invariance import train_test_invariance
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


def train_main(args):
    train_test_invariance.train_main(**vars(args))


def eval_main(args):
    train_test_invariance.eval_main(**vars(args))


def viz_main(args):
    train_test_invariance.viz_main(**vars(args))


def dim_viz_main(args):
    train_test_invariance.dim_viz_main(**vars(args))


@ros_init.with_ros("train_test_invariance")
def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    train_parser.add_argument('model_hparams', type=pathlib.Path)
    train_parser.add_argument('--checkpoint', type=pathlib.Path)
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--take', type=int)
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--log', '-l')
    train_parser.add_argument('--verbose', '-v', action='count', default=0)
    train_parser.add_argument('--log-scalars-every', type=int, help='loss/accuracy every this many steps/batches',
                              default=100)
    train_parser.add_argument('--validation-every', type=int, help='report validation every this many epochs',
                              default=1)
    train_parser.add_argument('--seed', type=int, default=None)
    train_parser.set_defaults(func=train_main)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    eval_parser.add_argument('checkpoint', type=pathlib.Path)
    eval_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val'], default='test')
    eval_parser.add_argument('--batch-size', type=int, default=32)
    eval_parser.add_argument('--verbose', '-v', action='count', default=0)
    eval_parser.add_argument('--seed', type=int, default=None)
    eval_parser.set_defaults(func=eval_main)

    viz_parser = subparsers.add_parser('viz')
    viz_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    viz_parser.add_argument('checkpoint', type=pathlib.Path)
    viz_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val'], default='test')
    viz_parser.add_argument('--verbose', '-v', action='count', default=0)
    viz_parser.add_argument('--seed', type=int, default=None)
    viz_parser.set_defaults(func=viz_main)

    dim_viz_parser = subparsers.add_parser('dim_viz')
    dim_viz_parser.add_argument('checkpoint', type=pathlib.Path)
    dim_viz_parser.set_defaults(func=dim_viz_main)

    args = parser.parse_args()

    if not hasattr(args, 'seed'):
        seed = None
    elif args.seed is None:
        seed = np.random.randint(0, 10000)
    else:
        seed = args.seed
    print("Using seed {}".format(seed))
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
