#!/usr/bin/env python
import argparse
import pathlib
import pickle

import numpy as np
import pyjacobian_follower

from arc_utilities import ros_init
from moveit.core import collision_detection, planning_scene


@ros_init.with_ros("generate_robot_pointcloud")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('outfilename', type=pathlib.Path)
    args = parser.parse_args()

    collision_sphere_link_name = 'collision_sphere'
    robot_points_generator = pyjacobian_follower.RobotPointsGenerator()
    links = robot_points_generator.get_link_names()
    points = {}
    res = 0.02
    for link_to_check in links:
        # returns the points of the collision in link frame
        points[link_to_check] = robot_points_generator.check_collision(link_to_check, res)

    with args.outfilename.open("wb") as outfile:
        pickle.dump(points, outfile)


if __name__ == "__main__":
    main()
