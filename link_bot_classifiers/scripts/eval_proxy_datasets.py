#!/usr/bin/env python
import argparse
import logging
import pathlib
from time import time

import numpy as np
import tensorflow as tf

from arc_utilities import ros_init
from link_bot_classifiers import train_test_classifier
from link_bot_classifiers.eval_proxy_datasets import eval_proxy_datasets
from link_bot_classifiers.train_test_classifier import add_eval_args

node_name = f"eval_proxy_datasets_{int(time())}"


@ros_init.with_ros(node_name)
def main():
    tf.get_logger().setLevel(logging.ERROR)
    np.set_printoptions(linewidth=250, precision=4, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints', type=pathlib.Path, nargs='+')
    add_eval_args(parser)

    args = parser.parse_args()

    eval_proxy_datasets(**vars(args))


if __name__ == '__main__':
    main()
