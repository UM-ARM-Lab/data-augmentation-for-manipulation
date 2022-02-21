#!/usr/bin/env python
from time import time

from arc_utilities import ros_init
from link_bot_pycommon.args import run_subparsers
from moonshine.magic import wandb_lightning_magic
from moonshine.torch_runner import make_arg_parsers
from state_space_dynamics import train_test_dynamics

node_name = f"train_test_propnet_{int(time())}"


@ros_init.with_ros(node_name)
def main():
    train_parser, viz_parser, eval_parser, eval_versions_parser, parser = make_arg_parsers(train_test_dynamics)

    # some overrides/additions
    train_parser.add_argument('--steps', type=int, default=25_000)

    wandb_lightning_magic()

    run_subparsers(parser)


if __name__ == '__main__':
    main()
