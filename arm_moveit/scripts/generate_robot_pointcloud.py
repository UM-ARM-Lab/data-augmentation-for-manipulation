#!/usr/bin/env python
import argparse
import pathlib
import pickle

import pyrobot_points_generator
from tqdm import tqdm

import rospy
from arc_utilities import ros_init


@ros_init.with_ros("generate_robot_pointcloud")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('outdir', type=pathlib.Path)
    parser.add_argument('--res', type=float, default=0.018)
    args = parser.parse_args()

    exclude_links = [
        'base_link', 'base_footprint', 'front_laser', 'front_lidar_mesh', 'hdt_michigan_root', 'base', 'pedestal_link',
        'rear_laser', 'rear_lidar_mesh', 'front_bumper_link', 'front_left_wheel_link', 'realsense',
        'front_right_wheel_link', 'imu_link', 'inertial_link', 'rear_bumper_link', 'rear_left_wheel_link',
        'rear_right_wheel_link', 'robot_payload', 'robot_root', 'top_chassis_link', 'top_plate_link',
        'top_plate_front_link', 'top_plate_rear_link', 'user_rail_link',
    ]

    args.outdir.mkdir(exist_ok=True, parents=True)
    outfilename = args.outdir / 'robot_points.pkl'
    if outfilename.exists():
        q = input(f"File {outfilename.as_posix()} already exist, do you want to overwrite? [Y/n]")
    if q == 'n':
        return

    robot_points_generator = pyrobot_points_generator.RobotPointsGenerator(args.res)
    links = robot_points_generator.get_link_names()
    points = {}
    for link_to_check in tqdm(links):
        # returns the points of the collision in link frame
        if link_to_check in exclude_links:
            continue

        p = robot_points_generator.check_collision(link_to_check)
        if len(p) > 0:
            points[link_to_check] = p

    data = {
        'robot_name': robot_points_generator.get_robot_name(),
        'res':        args.res,
        'points':     points,
    }

    with outfilename.open("wb") as outfile:
        pickle.dump(data, outfile)
    print(f"Wrote {outfilename.as_posix()}")


if __name__ == "__main__":
    main()
