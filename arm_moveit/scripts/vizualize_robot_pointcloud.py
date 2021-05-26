#!/usr/bin/env python
import argparse
import pathlib
import pickle
import random

import pyjacobian_follower
import tensorflow as tf

from arc_utilities import ros_init
from arc_utilities.listener import Listener
from link_bot_classifiers.robot_points import setup_robot_points, batch_transform_robot_points
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.moonshine_utils import repeat_tensor, numpify
from moonshine.simple_profiler import SimpleProfiler
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


def test_batched_perf():
    parser = argparse.ArgumentParser()
    parser.add_argument('robot_points', type=pathlib.Path, help='a pkl file')
    args = parser.parse_args()

    jacobian_follower = pyjacobian_follower.JacobianFollower('hdt_michigan', 0.001, True, True, False)

    with args.robot_points.open("rb") as file:
        data = pickle.load(file)

    points = data['points']
    link_names = list(points.keys())
    res = data['res']
    s = ScenarioWithVisualization()

    names = [
        "joint56",
        "joint57",
        "joint41",
        "joint42",
        "joint43",
        "joint44",
        "joint45",
        "joint46",
        "joint47",
        "leftgripper",
        "leftgripper2",
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "rightgripper",
        "rightgripper2",
    ]

    batch_size = 24
    points_per_links, points_link_frame_homo_batch = setup_robot_points(batch_size, points, link_names)

    positions = tf.random.normal([batch_size, 20])
    names = [names] * batch_size

    def _transform_robot_points():
        return batch_transform_robot_points(jacobian_follower,
                                            names,
                                            positions,
                                            points_per_links,
                                            points_link_frame_homo_batch,
                                            link_names)

    points_robot_frame_batch = _transform_robot_points()
    points_robot_frame_b = points_robot_frame_batch[0]

    points_robot_frame_b = numpify(points_robot_frame_b)
    s.plot_points_rviz(points_robot_frame_b, label='robot_points', frame_id='world', scale=res)

    p = SimpleProfiler()
    p.profile(100, _transform_robot_points)
    print(p)


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

    colors = get_link_colors(points)

    listener = Listener("/joint_states", JointState)
    while True:
        joint_state = listener.get()
        from time import perf_counter
        t0 = perf_counter()

        for link_name, link_points_link_frame in points.items():
            link_to_robot_transform = jacobian_follower.get_link_to_robot_transform(joint_state.name,
                                                                                    joint_state.position,
                                                                                    link_name)
            link_points_link_frame = tf.constant(link_points_link_frame, dtype=tf.float32)
            n_points = link_points_link_frame.shape[0]
            ones = tf.ones([n_points, 1], tf.float32)
            link_points_link_frame_homo = tf.concat([link_points_link_frame, ones], axis=1)
            link_points_link_frame_homo = tf.expand_dims(link_points_link_frame_homo)
            link_to_robot_transform = tf.cast(link_to_robot_transform, tf.float32)
            link_to_robot_transform_batch = repeat_tensor(link_to_robot_transform, n_points, 0, True)
            link_points_robot_frame = tf.matmul(link_to_robot_transform_batch, link_points_link_frame_homo)

            color = colors[link_name]
            link_points_robot_frame = numpify(link_points_robot_frame)
            s.plot_points_rviz(link_points_robot_frame, label=link_name, frame_id='world', scale=res, color=color)
        print(perf_counter() - t0)


def get_link_colors(points):
    colors = {}
    for link_name, link_points in points.items():
        r = random.random()
        b = random.random()
        g = random.random()
        colors[link_name] = (r, g, b)
    return colors


def test_matmul_perf():
    n = 10000

    def _a():
        x = tf.transpose(tf.matmul(tf.random.normal([4, 4]), tf.random.normal([4, n])))[:, :3]

    p = SimpleProfiler()
    p.profile(10000, _a)
    print(p)

    def _b():
        x = tf.matmul(tf.random.normal([1, 4, 4]), tf.random.normal([n, 4, 1]))[:, :3, 0]

    p = SimpleProfiler()
    p.profile(10000, _b)
    print(p)


@ros_init.with_ros("visualize_robot_point_cloud")
def main():
    """
    The viz functions assume that TF for the robot is running. The easiest ways to get this are to either
    1) run simulation
    2) run the real robot
    3) load the moveit config, run joint_state_publisher_gui, run robot_state_publisher
    """
    # viz_with_live_tf()
    # viz_with_internal_tf()
    test_batched_perf()
    # test_matmul_perf()


if __name__ == "__main__":
    main()
