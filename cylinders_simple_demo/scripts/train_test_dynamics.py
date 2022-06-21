#!/usr/bin/env python
import argparse
import pathlib

import numpy as np
import torch

from cylinders_simple_demo import train_test_propnet


def run_subparsers(parser: argparse.ArgumentParser):
    args = parser.parse_args()
    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


def train_main(args):
    train_test_propnet.train_main(
        dataset_dir=args.dataset_dir,
        model_params=args.model_params,
        nickname=args.nickname,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )


def eval_main(args):
    train_test_propnet.eval_main(
        dataset_dir=args.dataset_dir,
        checkpoint=args.checkpoint,
        mode=args.mode,
        batch_size=args.batch_size,
    )


def viz_main(args):
    train_test_propnet.viz_main(**vars(args))


def main():
    torch.set_printoptions(linewidth=250, precision=7, sci_mode=False)
    np.set_printoptions(linewidth=250, precision=7, suppress=True)
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('dataset_dir', type=pathlib.Path)
    train_parser.add_argument('model_params', type=pathlib.Path)
    train_parser.add_argument('--nickname', '-n', help='nickname')
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--epochs', type=int, default=500)
    train_parser.add_argument('--seed', type=int, default=0)
    train_parser.set_defaults(func=train_main)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('dataset_dir', type=pathlib.Path)
    eval_parser.add_argument('checkpoint')
    eval_parser.add_argument('--mode', type=str, default='test')
    eval_parser.add_argument('--batch-size', type=int, default=32)
    eval_parser.set_defaults(func=eval_main)

    viz_parser = subparsers.add_parser('viz')
    viz_parser.add_argument('dataset_dir', type=pathlib.Path)
    viz_parser.add_argument('checkpoint')
    viz_parser.add_argument('--mode', type=str, default='test')
    viz_parser.set_defaults(func=viz_main)

    run_subparsers(parser)


if __name__ == '__main__':
    main()
