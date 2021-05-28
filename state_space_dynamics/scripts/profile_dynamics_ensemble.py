#!/usr/bin/env python
import argparse
import pathlib

import hjson
import numpy as np

from arc_utilities import ros_init
from link_bot_classifiers.classifier_analysis_utils import predict, execute
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_pycommon.pycommon import make_dict_tf_float32
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.moonshine_utils import numpify, repeat
from state_space_dynamics import dynamics_utils
import tensorflow as tf


@ros_init.with_ros("test_dynamics")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="dataset", type=pathlib.Path)
    parser.add_argument("fwd_model_dir", help="load this saved forward model file", type=pathlib.Path, nargs='+')

    args = parser.parse_args()

    fwd_model, _ = dynamics_utils.load_generic_model(args.fwd_model_dir)

    dataset = DynamicsDatasetLoader([args.dataset_dir])
    tf_dataset = dataset.get_datasets(mode='train')
    inputs = next(iter(tf_dataset))

    for i in range(100):
        fwd_model.propagate_tf_batched(environment=environment,
                                       state=start_states,
                                       actions=expanded_actions)


if __name__ == '__main__':
    main()
