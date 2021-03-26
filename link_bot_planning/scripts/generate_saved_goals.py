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
from geometry_msgs.msg import Point, Pose
from link_bot_gazebo import gazebo_services
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.basic_3d_pose_marker import Basic3DPoseInteractiveMarker
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.get_scenario import get_scenario
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


@ros_init.with_ros("generate_saved_goals")
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
    parser.add_argument("--start-at", type=int, default=0)

    args = parser.parse_args()

    generate_saved_goals(method=args.method,
                         scenario=args.scenario,
                         n_trials=args.n_trials,
                         params_filename=args.params,
                         planner_params_filename=args.planner_params,
                         save_test_scenes_dir=args.scenes_dir,
                         start_at=args.start_at)


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

    goal_radius = planner_params['goal_params']['threshold']

    if method == 'rviz_marker':
        print("Run the following: rosrun rviz_visual_tools imarker_simple_demo")
        print("drag the marker to where you want and hit enter...")

    def make_marker(scale: float):
        marker = Marker(type=Marker.SPHERE)
        marker.scale = Point(2 * goal_radius, 2 * goal_radius, 2 * goal_radius)
        marker.color = ColorRGBA(0.5, 1.0, 0.5, 0.7)
        return marker

    goal_im = Basic3DPoseInteractiveMarker(make_marker=make_marker)

    for trial_idx in range(n_trials):
        if trial_idx < start_at:
            continue

        # restore
        bagfile_name = save_test_scenes_dir / f'scene_{trial_idx:04d}.bag'
        if not bagfile_name.exists():
            rospy.loginfo(Fore.YELLOW + f"No saved scene {bagfile_name}")
            continue

        rospy.loginfo(Fore.GREEN + f"Restoring scene {bagfile_name}")
        scenario.restore_from_bag(service_provider, planner_params, bagfile_name)
        environment = scenario.get_environment(planner_params)
        scenario.plot_environment_rviz(environment)

        current_goal = load(save_test_scenes_dir, trial_idx)
        if current_goal is not None:
            scenario.plot_goal_rviz(current_goal, goal_threshold=goal_radius)

        print(trial_idx)
        if method == 'rejection_sample':
            goal = rejection_sample_goal(scenario, params, planner_params, trial_idx)
        elif method == 'rviz_marker':
            if current_goal is not None:
                goal_im.set_pose(Pose(position=ros_numpy.msgify(Point, current_goal['point'])))
            goal = rviz_marker_goal(goal_im)
        else:
            raise NotImplementedError()

        save(save_test_scenes_dir, trial_idx, goal)


def rviz_marker_goal(goal_im: Basic3DPoseInteractiveMarker):
    input("press enter to save")
    pose = goal_im.get_pose()
    goal = {
        'point': ros_numpy.numpify(pose.position).astype(np.float32)
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


def load(save_test_scenes_dir: pathlib.Path, trial_idx: int):
    saved_goal_filename = save_test_scenes_dir / f'goal_{trial_idx:04d}.pkl'
    if saved_goal_filename.exists():
        with saved_goal_filename.open("rb") as saved_goal_file:
            goal = pickle.load(saved_goal_file)
        return goal
    else:
        return None


def save(save_test_scenes_dir: pathlib.Path, trial_idx: int, goal: Dict):
    saved_goal_filename = save_test_scenes_dir / f'goal_{trial_idx:04d}.pkl'
    rospy.loginfo(f"Saving goal to {saved_goal_filename}")
    with saved_goal_filename.open("wb") as saved_goal_file:
        pickle.dump(goal, saved_goal_file)


if __name__ == '__main__':
    main()
