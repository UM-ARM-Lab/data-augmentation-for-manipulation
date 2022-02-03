#!/usr/bin/env python
import argparse
import pathlib
from time import time

import numpy as np
import torch

from arc_utilities import ros_init
from augmentation import train_test_aug_vae
from link_bot_pycommon.args import run_subparsers
from moonshine.magic import wandb_lightning_magic


def train_main(args):
    if args.seed is None:
        args.seed = np.random.randint(0, 10000)

    train_test_aug_vae.train_main(**vars(args))


node_name = f"train_test_aug_vae_{int(time())}"


@ros_init.with_ros(node_name)
def main():
    torch.set_printoptions(linewidth=250, precision=7, sci_mode=False)
    np.set_printoptions(linewidth=250, precision=7, suppress=True)
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('dataset_dir', type=pathlib.Path)
    train_parser.add_argument('model_params_path', type=pathlib.Path)
    train_parser.add_argument('--nickname', '-n', type=str)
    train_parser.add_argument('--user', '-u', type=str, default='armlab')
    train_parser.add_argument('--checkpoint')
    train_parser.add_argument('--batch-size', type=int, default=60)
    train_parser.add_argument('--take', type=int)
    train_parser.add_argument('--epochs', type=int, default=-1)
    train_parser.add_argument('--steps', type=int, default=5_000)
    train_parser.add_argument('--no-validate', action='store_true')
    train_parser.add_argument('--seed', type=int, default=None)
    train_parser.set_defaults(func=train_main)

    wandb_lightning_magic()

    run_subparsers(parser)


if __name__ == '__main__':
    main()
