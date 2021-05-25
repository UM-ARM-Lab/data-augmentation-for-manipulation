#!/usr/bin/env python
import argparse
import pathlib
import pickle

import pyrobot_points_generator

import rospy
from arc_utilities import ros_init
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization


@ros_init.with_ros("visualize_robot_point_cloud")
def main():
    """
    This script assumes that TF for the robot is running. The easiest ways to get this are to either
    1) run simulation
    2) run the real robot
    3) load the moveit config, run joint_state_publisher_gui, run robot_state_publisher
    """
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
            print(link_name)
            if len(link_points) > 0:
                s.plot_points_rviz(link_points, label=link_name, frame_id=link_name, scale=res)


if __name__ == "__main__":
    main()
