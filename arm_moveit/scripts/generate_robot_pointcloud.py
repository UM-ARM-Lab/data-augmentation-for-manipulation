#!/usr/bin/env python
import argparse
import pathlib
import pickle

import pyrobot_points_generator

import rospy
from arc_utilities import ros_init


@ros_init.with_ros("generate_robot_pointcloud")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('outdir', type=pathlib.Path)
    args = parser.parse_args()

    res = 0.02
    robot_points_generator = pyrobot_points_generator.RobotPointsGenerator(res)
    links = robot_points_generator.get_link_names()
    points = {}
    for link_to_check in links:
        # returns the points of the collision in link frame
        points[link_to_check] = robot_points_generator.check_collision(link_to_check)

    args.outdir.mkdir(exist_ok=True, parents=True)
    outfilename = args.outdir / 'robot_points.pkl'
    with outfilename.open("wb") as outfile:
        pickle.dump(points, outfile)

    rospy.sleep(1)


if __name__ == "__main__":
    main()
