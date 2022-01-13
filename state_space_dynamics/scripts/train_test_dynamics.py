#!/usr/bin/env python
import argparse
import logging
import pathlib
from time import time

import numpy as np
import tensorflow

from arc_utilities import ros_init
from moonshine.gpu_config import limit_gpu_mem
from state_space_dynamics import train_test_dynamics

limit_gpu_mem(None)


def train_main(args, seed: int):
    train_test_dynamics.train_main(dataset_dirs=args.dataset_dirs,
                                   model_hparams=args.model_hparams,
                                   checkpoint=args.checkpoint,
                                   log=args.log,
                                   batch_size=args.batch_size,
                                   epochs=args.epochs,
                                   seed=seed,
                                   ensemble_idx=args.ensemble_idx,
                                   trials_directory=pathlib.Path('trials'),
                                   take=args.take,
                                   )


def eval_main(args, seed: int):
    train_test_dynamics.eval_main(args.dataset_dirs, args.checkpoint, args.mode, args.batch_size)


def viz_main(args, seed: int):
    train_test_dynamics.viz_main(**vars(args))


now = str(int(time()))
node_name = f"train_test_{now}"


@ros_init.with_ros(node_name)
def main():
    parser = argparse.ArgumentParser()
    tensorflow.get_logger().setLevel(logging.ERROR)

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    train_parser.add_argument('model_hparams', type=pathlib.Path)
    train_parser.add_argument('--checkpoint', type=pathlib.Path)
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--take', type=int)
    train_parser.add_argument('--epochs', type=int, default=500)
    train_parser.add_argument('--ensemble-idx', type=int)
    train_parser.add_argument('--log', '-l')
    train_parser.add_argument('--verbose', '-v', action='count', default=0)
    train_parser.add_argument('--log-scalars-every', type=int, help='loss/accuracy every this many steps/batches',
                              default=100)
    train_parser.add_argument('--validation-every', type=int, help='report validation every this many epochs',
                              default=1)
    train_parser.add_argument('--seed', type=int, default=None)
    train_parser.add_argument('--use-gt-rope', action='store_true')
    train_parser.set_defaults(func=train_main)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    eval_parser.add_argument('checkpoint', type=pathlib.Path)
    eval_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val', 'all'], default='test')
    eval_parser.add_argument('--batch-size', type=int, default=32)
    eval_parser.add_argument('--verbose', '-v', action='count', default=0)
    eval_parser.add_argument('--seed', type=int, default=None)
    eval_parser.add_argument('--use-gt-rope', action='store_true')
    eval_parser.set_defaults(func=eval_main)

    viz_parser = subparsers.add_parser('viz')
    viz_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    viz_parser.add_argument('checkpoint', type=pathlib.Path)
    viz_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val', 'all'], default='test')
    viz_parser.add_argument('--verbose', '-v', action='count', default=0)
    viz_parser.add_argument('--seed', type=int, default=None)
    viz_parser.add_argument('--use-gt-rope', action='store_true')
    viz_parser.set_defaults(func=viz_main)

    args = parser.parse_args()

    if args.seed is None:
        seed = np.random.randint(0, 10000)
    else:
        seed = args.seed
    print("Using seed {}".format(seed))

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args, seed)


if __name__ == '__main__':
    main()
