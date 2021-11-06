#!/usr/bin/env python
import argparse
import pathlib

from arc_utilities import ros_init
from dm_envs.add_velocity_to_dynamics_dataset import add_velocity_to_dataset


@ros_init.with_ros("add_vel")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')

    args = parser.parse_args()

    add_velocity_to_dataset(args.dataset_dir)


if __name__ == '__main__':
    main()
