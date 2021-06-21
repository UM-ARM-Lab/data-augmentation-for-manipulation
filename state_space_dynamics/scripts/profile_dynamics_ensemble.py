#!/usr/bin/env python
import argparse
import pathlib

from progressbar import progressbar

from arc_utilities import ros_init
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_pycommon.get_scenario import get_scenario
from state_space_dynamics import dynamics_utils


@ros_init.with_ros("profile_dynamics_ensemble")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="dataset", type=pathlib.Path)
    parser.add_argument("fwd_model_dir", help="load this saved forward model file", type=pathlib.Path)

    args = parser.parse_args()

    scenario = get_scenario('dual_arm_rope_sim_val_with_robot_feasibility_checking')
    fwd_model = dynamics_utils.load_generic_model(args.fwd_model_dir, scenario=scenario)

    dataset = DynamicsDatasetLoader([args.dataset_dir])
    tf_dataset = dataset.get_datasets(mode='train')
    b = 64
    tf_dataset = tf_dataset.batch(b)
    inputs = tf_dataset.get_element(0)

    for i in progressbar(range(10)):
        for short_inputs in dataset.split_into_sequences(inputs, 2, time_dim=1):
            fwd_model.propagate_from_example(short_inputs)


if __name__ == '__main__':
    main()
