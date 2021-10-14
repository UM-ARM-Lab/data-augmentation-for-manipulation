#!/usr/bin/env python
import argparse
import logging
import pathlib
from time import time

import tensorflow as tf

from arc_utilities import ros_init
from link_bot_classifiers.augment_classifier_dataset import augment_classifier_dataset
from link_bot_data.new_classifier_dataset import NewClassifierDatasetLoader
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_hjson
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


@ros_init.with_ros("augment_classifier_dataset")
def main():
    tf.get_logger().setLevel(logging.FATAL)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('--n-augmentations', type=int, default=10)
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    suffix = f"aug-{int(time())}"
    dataset_dir = args.dataset_dir

    outdir = dataset_dir.parent / f"{dataset_dir.name}+{suffix}"

    scenario = get_scenario("dual_arm_rope_sim_val_with_robot_feasibility_checking")
    hparams = load_hjson(pathlib.Path("hparams/classifier/aug.hjson"))

    augment_classifier_dataset(dataset_dir=dataset_dir,
                               hparams=hparams,
                               outdir=outdir,
                               n_augmentations=args.n_augmentations,
                               scenario=scenario)


if __name__ == '__main__':
    main()
