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

    args.outdir.mkdir(exist_ok=True, parents=True)
    outfilename = args.outdir / 'robot_points.pkl'
    if outfilename.exists():
        q = input(f"File {outfilename.as_posix()} already exist, do you want to overwrite? [Y/n]")
        if q == 'n':
            return

    res = 0.02
    robot_points_generator = pyrobot_points_generator.RobotPointsGenerator(res)
    links = robot_points_generator.get_link_names()
    points = {}
    for link_to_check in links:
        # returns the points of the collision in link frame
        p = robot_points_generator.check_collision(link_to_check)
        if len(p) > 0:
            points[link_to_check] = p

    data = {
        'robot_name': robot_points_generator.get_robot_name(),
        'res':        res,
        'points':     points,
    }

    with outfilename.open("wb") as outfile:
        pickle.dump(data, outfile)
    print(f"Wrote {outfilename.as_posix()}")


if __name__ == "__main__":
    main()
