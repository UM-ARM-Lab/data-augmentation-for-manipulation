#!/usr/bin/env python
import argparse
import pathlib
from typing import Optional, Dict, List

import numpy as np
from colorama import Fore

import ros_numpy
import rospy
from arc_utilities import ros_init
from arm_robots.robot import RobotPlanningError
from geometry_msgs.msg import Point, Pose
from link_bot_gazebo import gazebo_services
from link_bot_gazebo.gazebo_utils import get_gazebo_processes
from link_bot_planning.test_scenes import get_states_to_save, save_test_scene, get_all_scene_indices, \
    make_goal_filename, load_goal, save_goal
from link_bot_pycommon.args import int_set_arg
from link_bot_pycommon.basic_3d_pose_marker import Basic3DPoseInteractiveMarker
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.pycommon import deal_with_exceptions
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


@ros_init.with_ros("generate_saved_goals")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=str, help='scenario')
    parser.add_argument("scenes_dir", type=pathlib.Path)
    parser.add_argument("method", type=str, choices=['rejection_sample', 'rviz_marker'])
    parser.add_argument("--trials", type=int_set_arg, default=None)

    args = parser.parse_args()

    generate_saved_goals(method=args.method,
                         scenario=args.scenario,
                         trials=args.trials,
                         scenes_dir=args.scenes_dir)


def generate_saved_goals(method: str,
                         scenario: str,
                         trials: Optional[List[int]],
                         scenes_dir: Optional[pathlib.Path] = None
                         ):
    scenes_dir.mkdir(exist_ok=True, parents=True)

    [p.resume() for p in get_gazebo_processes()]

    scenario = get_scenario(scenario)
    service_provider = gazebo_services.GazeboServices()
    service_provider.setup_env()

    params = sketchy_default_params()

    scenario.move_objects_out_of_scene(params)
    scenario.on_before_data_collection(params)
    scenario.randomization_initialization(params)

    goal_radius = params['goal_params']['threshold']

    if method == 'rviz_marker':
        print("drag the marker to where you want and hit enter...")

    def make_marker(scale: float):
        marker = Marker(type=Marker.SPHERE)
        marker.scale = Point(2 * goal_radius, 2 * goal_radius, 2 * goal_radius)
        marker.color = ColorRGBA(0.5, 1.0, 0.5, 0.7)
        return marker

    goal_im = Basic3DPoseInteractiveMarker(make_marker=make_marker)

    if trials is None:
        trials = get_all_scene_indices(scenes_dir)

    for trial_idx in trials:
        # restore
        bagfile_name = scenes_dir / f'scene_{trial_idx:04d}.bag'
        if bagfile_name.exists():
            rospy.loginfo(Fore.GREEN + f"Restoring scene {bagfile_name}" + Fore.RESET)

            def _restore():
                scenario.restore_from_bag(service_provider, params, bagfile_name)

            def _retry_msg():
                input("failed to plan, try to move the obstacles out of the way first")

            deal_with_exceptions(how_to_handle='retry',
                                 function=_restore,
                                 exception_callback=_retry_msg,
                                 exceptions=(RobotPlanningError,))

        environment = scenario.get_environment(params)
        for _ in range(3):
            scenario.plot_environment_rviz(environment)
            rospy.sleep(0.1)

        current_goal = load(scenes_dir, trial_idx)
        state = scenario.get_state()
        scenario.reset_viz()
        scenario.plot_state_rviz(state, label='start')
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

        save_test_scene(joint_state, links_states, scenes_dir, trial_idx, force=True)
        save_goal(goal, scenes_dir, trial_idx)

    scenario.robot.disconnect()


def sketchy_default_params():
    params = {
        'environment_randomization': {
            'type':          'jitter',
            'nominal_poses': {
            },
        },
        'reset_joint_config':        {
            'joint1':  1.5,
            'joint2':  0.00019175345369149,
            'joint3':  0.06807247549295425,
            'joint4':  -1.0124582052230835,
            'joint41': 1.5,
            'joint42': 0.0,
            'joint43': 0.1562790721654892,
            'joint44': -0.9140887260437012,
            'joint45': 1.601524829864502,
            'joint46': -1.2170592546463013,
            'joint47': -0.016490796580910683,
            'joint56': 0.0,
            'joint57': -0.3,
            'joint5':  -1.5196460485458374,
            'joint6':  1.3154287338256836,
            'joint7':  -0.01706605777144432,
        },
        'res':                       0.02,
        'extent':                    [-0.6, 0.6, 0.25, 1.15, -0.3, 0.6],
        'goal_params':               {
            'threshold': 0.045,
        }
    }
    params['real_val_rope_reset_joint_config'] = params['reset_joint_config']
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
    saved_goal_filename = make_goal_filename(save_test_scenes_dir, trial_idx)
    if saved_goal_filename.exists():
        return load_goal(save_test_scenes_dir, trial_idx)
    else:
        return None


if __name__ == '__main__':
    main()
