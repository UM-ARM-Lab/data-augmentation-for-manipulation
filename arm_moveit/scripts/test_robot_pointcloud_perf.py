#!/usr/bin/env python
import tensorflow as tf
import argparse
import pathlib
import pickle

import pyrobot_points_generator

import rospy
from arc_utilities import ros_init
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization


def transform_robot_pointcloud(points: Dict, joint_positions):
    points_robot_frame = tf.matmul(transforms, points_link_frame)
    return points_robot_frame


@ros_init.with_ros("test_robot_point_cloud_perf")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('robot_points', type=pathlib.Path, help='a pkl file')
    args = parser.parse_args()

    with args.robot_points.open("rb") as file:
        data = pickle.load(file)

    points = data['points']
    res = data['res']
    joint_positions = tf.zeros(20)
    from time import perf_counter
    t0 = perf_counter()
    points_robot_frame = transform_robot_pointcloud(points, joint_positions)
    print(perf_counter() - t0)

    s = ScenarioWithVisualization()
    s.plot_points_rviz(points_robot_frame.numpy(), 'transformed_points', frame_id='robot_root', scale=res)


if __name__ == "__main__":
    main()
