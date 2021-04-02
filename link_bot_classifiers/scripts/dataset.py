#!/usr/bin/env python

import argparse
import pathlib

import colorama

from arc_utilities import ros_init
from link_bot_data.classifier_dataset import ClassifierDatasetLoader


def print_main(args):
    dataset = ClassifierDatasetLoader(args.dataset_dirs, load_true_states=True)
    dataset.pprint_example()


@ros_init.with_ros('dataset')
def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()
    print_parser = subparsers.add_parser('print')
    print_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    print_parser.set_defaults(func=print_main)

    args = parser.parse_args()

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
