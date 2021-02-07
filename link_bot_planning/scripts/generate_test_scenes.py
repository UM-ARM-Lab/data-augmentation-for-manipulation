#!/usr/bin/env python
import argparse
import logging
import pathlib
from typing import Optional

import colorama
import hjson
import tensorflow as tf

from arc_utilities import ros_init
from link_bot_gazebo_python import gazebo_services
from link_bot_planning.test_scenes import save_test_scene, create_randomized_start_state, get_states_to_save
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_scenario import get_scenario


def main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("scenario", type=str, help='scenario')
    parser.add_argument("params", type=pathlib.Path, help='the data collection params file should work')
    parser.add_argument("scenes_dir", type=pathlib.Path)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--test-restore", action='store_true')
    parser.add_argument("--start-at", type=int, default=0)

    args = parser.parse_args()

    ros_init.rospy_and_cpp_init('generate_test_scenes')
    generate_test_scenes(scenario=args.scenario,
                         n_trials=args.n_trials,
                         params_filename=args.params,
                         test_restore=args.test_restore,
                         save_test_scenes_dir=args.scenes_dir,
                         start_at=args.start_at)
    ros_init.shutdown()


def generate_test_scenes(scenario: str,
                         n_trials: int,
                         params_filename: pathlib.Path,
                         start_at: int,
                         test_restore: bool,
                         save_test_scenes_dir: Optional[pathlib.Path] = None,
                         ):
    save_test_scenes_dir.mkdir(exist_ok=True, parents=True)

    service_provider = gazebo_services.GazeboServices()
    scenario = get_scenario(scenario)
    scenario.robot.raise_on_failure = False

    service_provider.setup_env(verbose=0,
                               real_time_rate=0.0,
                               max_step_size=0.01,
                               play=True)

    with params_filename.open("r") as params_file:
        params = hjson.load(params_file)

    scenario.on_before_data_collection(params)
    scenario.randomization_initialization(params)

    for trial_idx in range(n_trials):
        if trial_idx < start_at:
            continue

        create_randomized_start_state(params, scenario, trial_idx)

        joint_state, links_states = get_states_to_save()

        bagfile_name = save_test_scene(joint_state, links_states, save_test_scenes_dir, trial_idx)

        if test_restore:
            scenario.robot.plan_to_joint_config("both_arms", 'home')
            joint_config = {}
            # NOTE: this will not work on victor because grippers don't work the same way
            for joint_name in scenario.robot.get_move_group_commander("whole_body").get_active_joints():
                index_of_joint_name_in_state_msg = joint_state.name.index(joint_name)
                joint_config[joint_name] = joint_state.position[index_of_joint_name_in_state_msg]
            scenario.robot.plan_to_joint_config("whole_body", joint_config)
            service_provider.pause()
            service_provider.restore_from_bag(bagfile_name=bagfile_name, excluded_models=['hdt_michigan'])
            service_provider.play()


if __name__ == '__main__':
    main()
