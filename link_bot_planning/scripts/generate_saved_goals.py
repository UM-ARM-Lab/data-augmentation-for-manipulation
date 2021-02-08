#!/usr/bin/env python
import argparse
import logging
import pathlib
import pickle
from typing import Optional, Dict

import colorama
import hjson
import numpy as np
import tensorflow as tf
from colorama import Fore

import ros_numpy
import rospy
from arc_utilities import ros_init
from arc_utilities.listener import Listener
from link_bot_gazebo_python import gazebo_services
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.get_scenario import get_scenario
from visualization_msgs.msg import InteractiveMarkerFeedback


def main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("scenario", type=str, help='scenario')
    parser.add_argument("params", type=pathlib.Path, help='the data collection params file should work')
    parser.add_argument("planner_params", type=pathlib.Path, help='the planner common params file should work')
    parser.add_argument("scenes_dir", type=pathlib.Path)
    parser.add_argument("method", type=str, choices=['rejection_sample', 'rviz_marker'])
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--test-restore", action='store_true')
    parser.add_argument("--start-at", type=int, default=0)

    args = parser.parse_args()

    ros_init.rospy_and_cpp_init('generate_test_scenes')
    generate_saved_goals(method=args.method,
                         scenario=args.scenario,
                         n_trials=args.n_trials,
                         params_filename=args.params,
                         planner_params_filename=args.planner_params,
                         save_test_scenes_dir=args.scenes_dir,
                         start_at=args.start_at)
    ros_init.shutdown()


def generate_saved_goals(method: str,
                         scenario: str,
                         n_trials: int,
                         params_filename: pathlib.Path,
                         planner_params_filename: pathlib.Path,
                         start_at: int,
                         save_test_scenes_dir: Optional[pathlib.Path] = None
                         ):
    save_test_scenes_dir.mkdir(exist_ok=True, parents=True)

    service_provider = gazebo_services.GazeboServices()
    scenario = get_scenario(scenario)
    service_provider.setup_env(verbose=0,
                               real_time_rate=0.0,
                               max_step_size=0.01,
                               play=True)
    with params_filename.open("r") as params_file:
        params = hjson.load(params_file)
    with planner_params_filename.open("r") as planner_params_file:
        planner_params = hjson.load(planner_params_file)
    scenario.on_before_data_collection(params)
    scenario.randomization_initialization(params)

    if method == 'rviz_marker':
        print("Run the following: rosrun rviz_visual_tools imarker_simple_demo")
        print("drag the marker to where you want and hit enter...")

    for trial_idx in range(n_trials):
        if trial_idx < start_at:
            continue

        # restore
        bagfile_name = save_test_scenes_dir / f'scene_{trial_idx:04d}.bag'
        rospy.loginfo(Fore.GREEN + f"Restoring scene {bagfile_name}")
        scenario.restore_from_bag(service_provider, planner_params, bagfile_name)

        if method == 'rejection_sample':
            goal = rejection_sample_goal(scenario, params, planner_params, trial_idx)
        elif method == 'rviz_marker':
            goal = rviz_marker_goal(scenario, params, planner_params, trial_idx)
        else:
            raise NotImplementedError()

        save(save_test_scenes_dir, trial_idx, goal)


def rviz_marker_goal(scenario: ExperimentScenario, params: Dict, planner_params: Dict, trial_idx: int):
    listener = Listener('/imarker_simple_demo/imarker/feedback', InteractiveMarkerFeedback)
    input('press enter to save')
    feedback: InteractiveMarkerFeedback = listener.get()
    goal = {
        'point': ros_numpy.numpify(feedback.pose.position).astype(np.float32)
    }
    return goal


def rejection_sample_goal(scenario: ExperimentScenario, params: Dict, planner_params: Dict, trial_idx: int):
    goal_rng = np.random.RandomState(trial_idx)
    while True:
        env = scenario.get_environment(params)
        goal = scenario.sample_goal(environment=env, rng=goal_rng, planner_params=planner_params)
        scenario.plot_goal_rviz(goal, goal_threshold=planner_params['goal_params']['threshold'])
        if input("ok? [Y/n]") in ['y', 'Y', 'yes', '']:
            break

    return goal


def save(save_test_scenes_dir: pathlib.Path, trial_idx: int, goal: Dict):
    saved_goal_filename = save_test_scenes_dir / f'goal_{trial_idx:04d}.pkl'
    rospy.loginfo(f"Saving goal to {saved_goal_filename}")
    with saved_goal_filename.open("wb") as saved_goal_file:
        pickle.dump(goal, saved_goal_file)


if __name__ == '__main__':
    main()
