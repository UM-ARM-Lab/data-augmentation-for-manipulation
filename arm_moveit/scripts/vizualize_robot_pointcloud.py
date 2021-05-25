#!/usr/bin/env python
import argparse
import pathlib
import pickle
import random

import pyjacobian_follower
import tensorflow as tf

from arc_utilities import ros_init
from arc_utilities.listener import Listener
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.moonshine_utils import repeat_tensor, numpify
from sensor_msgs.msg import JointState


def viz_with_live_tf():
    parser = argparse.ArgumentParser()
    parser.add_argument('robot_points', type=pathlib.Path, help='a pkl file')
    args = parser.parse_args()

    with args.robot_points.open("rb") as file:
        data = pickle.load(file)

    points = data['points']
    res = data['res']
    s = ScenarioWithVisualization()
    while True:
        for link_name, link_points in points.items():
            s.plot_points_rviz(link_points, label=link_name, frame_id=link_name, scale=res)


def viz_with_internal_tf():
    parser = argparse.ArgumentParser()
    parser.add_argument('robot_points', type=pathlib.Path, help='a pkl file')
    args = parser.parse_args()

    jacobian_follower = pyjacobian_follower.JacobianFollower('hdt_michigan', 0.001, True, True, False)

    with args.robot_points.open("rb") as file:
        data = pickle.load(file)

    points = data['points']
    res = data['res']
    s = ScenarioWithVisualization()

    colors = {}
    for link_name, link_points in points.items():
        r = random.random()
        b = random.random()
        g = random.random()
        colors[link_name] = (r, g, b)

    listener = Listener("/joint_states", JointState)
    while True:
        joint_state = listener.get()
        from time import perf_counter
        t0 = perf_counter()
        for link_name, link_points_link_frame in points.items():
            link_points_link_frame = tf.constant(link_points_link_frame, dtype=tf.float32)
            link_points_link_frame_homo = tf.expand_dims(tf.concat([link_points_link_frame, tf.ones([link_points_link_frame.shape[0], 1], tf.float32)], axis=1), axis=2)
            link_to_robot_transform = jacobian_follower.get_link_to_robot_transform(joint_state.name,
                                                                                    joint_state.position,
                                                                                    link_name)
            link_to_robot_transform = tf.cast(link_to_robot_transform, tf.float32)
            link_to_robot_transform_batch = repeat_tensor(link_to_robot_transform, link_points_link_frame.shape[0], 0, True)
            link_points_robot_frame = tf.matmul(link_to_robot_transform_batch, link_points_link_frame_homo)

            color = colors[link_name]
            link_points_robot_frame = numpify(link_points_robot_frame)
            s.plot_points_rviz(link_points_robot_frame, label=link_name, frame_id='world', scale=res, color=color)
        print(perf_counter() - t0)


@ros_init.with_ros("visualize_robot_point_cloud")
def main():
    """
    This script assumes that TF for the robot is running. The easiest ways to get this are to either
    1) run simulation
    2) run the real robot
    3) load the moveit config, run joint_state_publisher_gui, run robot_state_publisher
    """
    # viz_with_live_tf()
    viz_with_internal_tf()


if __name__ == "__main__":
    main()
