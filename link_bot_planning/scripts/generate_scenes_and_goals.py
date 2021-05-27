#!/usr/bin/env python
import argparse
import logging
import pathlib
import pickle
from typing import Optional, Dict, List

import colorama
import numpy as np
import tensorflow as tf
from colorama import Fore

import ros_numpy
import rospy
from arc_utilities import ros_init
from arm_robots.robot import RobotPlanningError
from geometry_msgs.msg import Point, Pose
from link_bot_gazebo import gazebo_services
from link_bot_planning.test_scenes import get_states_to_save, save_test_scene
from link_bot_pycommon.args import my_formatter, int_set_arg
from link_bot_pycommon.basic_3d_pose_marker import Basic3DPoseInteractiveMarker
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.pycommon import deal_with_exceptions
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


@ros_init.with_ros("generate_saved_goals")
def main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("scenario", type=str, help='scenario')
    parser.add_argument("scenes_dir", type=pathlib.Path)
    parser.add_argument("method", type=str, choices=['rejection_sample', 'rviz_marker'])
    parser.add_argument("--trials", type=int_set_arg, default=100)

    args = parser.parse_args()

    generate_saved_goals(method=args.method,
                         scenario=args.scenario,
                         trials=args.trials,
                         save_test_scenes_dir=args.scenes_dir)


def generate_saved_goals(method: str,
                         scenario: str,
                         trials: List[int],
                         save_test_scenes_dir: Optional[pathlib.Path] = None
                         ):
    save_test_scenes_dir.mkdir(exist_ok=True, parents=True)

    service_provider = gazebo_services.GazeboServices()
    scenario = get_scenario(scenario)
    service_provider.setup_env(verbose=0,
                               real_time_rate=0.0,
                               max_step_size=0.01,
                               play=True)

    params = get_params()

    scenario.move_objects_out_of_scene(params)
    scenario.on_before_data_collection(params)
    scenario.randomization_initialization(params)

    goal_radius = params['goal_params']['threshold']

    if method == 'rviz_marker':
        print("Run the following: rosrun rviz_visual_tools imarker_simple_demo")
        print("drag the marker to where you want and hit enter...")

    def make_marker(scale: float):
        marker = Marker(type=Marker.SPHERE)
        marker.scale = Point(2 * goal_radius, 2 * goal_radius, 2 * goal_radius)
        marker.color = ColorRGBA(0.5, 1.0, 0.5, 0.7)
        return marker

    goal_im = Basic3DPoseInteractiveMarker(make_marker=make_marker)

    for trial_idx in trials:
        # restore
        bagfile_name = save_test_scenes_dir / f'scene_{trial_idx:04d}.bag'
        if bagfile_name.exists():
            rospy.loginfo(Fore.GREEN + f"Restoring scene {bagfile_name}")

            def _restore():
                scenario.restore_from_bag(service_provider, params, bagfile_name)

            def _retry_msg():
                input("failed to plan, try to move the obstacles out of the way first")

            deal_with_exceptions(how_to_handle='retry',
                                 function=_restore,
                                 exception_callback=_retry_msg,
                                 exceptions=(RobotPlanningError,))

        environment = scenario.get_environment(params)
        scenario.plot_environment_rviz(environment)

        current_goal = load(save_test_scenes_dir, trial_idx)
        if current_goal is not None:
            scenario.plot_goal_rviz(current_goal, goal_threshold=goal_radius)

        print(trial_idx)
        if method == 'rejection_sample':
            goal = rejection_sample_goal(scenario, params, params, trial_idx)
        elif method == 'rviz_marker':
            if current_goal is not None:
                goal_im.set_pose(Pose(position=ros_numpy.msgify(Point, current_goal['point'])))
            input("press enter to save")
            goal = rviz_marker_goal(goal_im)
        else:
            raise NotImplementedError()

        rospy.loginfo(f"Saving {trial_idx}")
        joint_state, links_states = get_states_to_save()

        save_test_scene(joint_state, links_states, save_test_scenes_dir, trial_idx, force=True)

        save(save_test_scenes_dir, trial_idx, goal)


def get_params():
    params = {
        'environment_randomization': {
            'type':          'jitter',
            'nominal_poses': {
                'long_hook1': None,
                'wall2':      None,
            },
        },
        'reset_joint_config':        {
            'joint1':  0,
            'joint2':  0,
            'joint3':  0,
            'joint4':  0,
            'joint41': 0,
            'joint42': 0,
            'joint43': 0,
            'joint44': 0,
            'joint45': 0,
            'joint46': 0,
            'joint47': 0,
            'joint5':  0,
            'joint56': 0,
            'joint57': 0,
            'joint6':  0,
            'joint7':  0,
        },
        'res':                       0.02,
        'extent':                    [-0.6, 0.6, 0.25, 1.15, -0.3, 0.6],
        'goal_params':               {
            'threshold': 0.05,
        }
    }
    return params


def rviz_marker_goal(goal_im: Basic3DPoseInteractiveMarker):
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
    with saved_goal_filename.open("wb") as saved_goal_file:
        pickle.dump(goal, saved_goal_file)


if __name__ == '__main__':
    main()
