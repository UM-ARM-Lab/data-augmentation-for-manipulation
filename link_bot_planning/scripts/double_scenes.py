#!/usr/bin/env python
import argparse
import logging
import pathlib

import colorama
import tensorflow as tf


def main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("dirname", type=pathlib.Path)

    args = parser.parse_args()

    n_existing_scenes = len(list(args.dirname.glob("*.bag")))

    for i in range(n_existing_scenes):
        j = i + n_existing_scenes
        print(f'{i}->{j}')


if __name__ == '__main__':
    main()
