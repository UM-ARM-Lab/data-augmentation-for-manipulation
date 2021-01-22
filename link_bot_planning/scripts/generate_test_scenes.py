#!/usr/bin/env python
import argparse
import logging
import pathlib
from typing import Optional

import colorama
import hjson
import numpy as np
import tensorflow as tf

import rosbag
import rospy
from arc_utilities.listener import Listener
from gazebo_msgs.msg import LinkStates
from link_bot_gazebo_python import gazebo_services
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_scenario import get_scenario
from sensor_msgs.msg import JointState


def main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("scenario", type=str, help='scenario')
    parser.add_argument("params", type=pathlib.Path, help='the data collection params file should work')
    parser.add_argument("scenes_dir", type=pathlib.Path)
    parser.add_argument("--n-trials", type=int, default=100)

    args = parser.parse_args()

    rospy.init_node("save_test_scenes")

    return generate_test_scenes(scenario=args.scenario,
                                n_trials=args.n_trials,
                                params_filename=args.params,
                                save_test_scenes_dir=args.scenes_dir)


def generate_test_scenes(scenario: str,
                         n_trials: int,
                         params_filename: pathlib.Path,
                         save_test_scenes_dir: Optional[pathlib.Path] = None,
                         ):
    service_provider = gazebo_services.GazeboServices()
    scenario = get_scenario(scenario)

    service_provider.setup_env(verbose=0,
                               real_time_rate=0.0,
                               max_step_size=0.01,
                               play=True)

    link_states_listener = Listener("gazebo/link_states", LinkStates)

    env_rng = np.random.RandomState(0)
    action_rng = np.random.RandomState(0)

    with params_filename.open("r") as params_file:
        params = hjson.load(params_file)

    scenario.on_before_data_collection(params)
    scenario.randomization_initialization(params)

    for trial_idx in range(n_trials):
        environment = scenario.get_environment(params)

        scenario.randomize_environment(env_rng, params)

        for i in range(10):
            state = scenario.get_state()
            action = scenario.sample_action(action_rng=action_rng,
                                            environment=environment,
                                            state=state,
                                            action_params=params,
                                            validate=True,
                                            )
            scenario.execute_action(action)

        links_states = link_states_listener.get()
        joint_state = scenario.robot._joint_state_listener.get()
        save_test_scenes_dir.mkdir(exist_ok=True, parents=True)
        bagfile_name = save_test_scenes_dir / f'scene_{trial_idx:04d}.bag'
        rospy.loginfo(f"Saving scene to {bagfile_name}")
        with rosbag.Bag(bagfile_name, 'w') as bag:
            bag.write('links_states', links_states)
            bag.write('joint_state', joint_state)


if __name__ == '__main__':
    main()
