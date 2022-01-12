#!/usr/bin/env python
import argparse
import csv
import pathlib
import re

import numpy as np
import transformations

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper


def main():
    rospy.init_node("colmap_calibrate")

    parser = argparse.ArgumentParser()
    parser.add_argument('indir', type=pathlib.Path, help='colmap workspace, also contains transforms dir')

    args = parser.parse_args()

    images_txt = args.indir / 'sparse' / '0' / 'images.txt'
    transforms_dir = args.indir / 'transforms'
    transforms_filenames = transforms_dir.glob("*.txt")
    transforms_to_frame_id_map = {}
    for transforms_filename in transforms_filenames:
        m = re.match(r'.*transform(\d+)\.txt', transforms_filename.as_posix())
        frame_id = int(m.group(1))
        transform = np.loadtxt(transforms_filename.as_posix())
        transforms_to_frame_id_map[frame_id] = transform

    colmap_poses = []
    mocap_poses = []
    with images_txt.open("r") as f:
        reader = csv.reader(f, delimiter=' ')
        next(reader)
        next(reader)
        next(reader)
        next(reader)
        while True:
            try:
                l1 = next(reader)
                image_filename = l1[-1]
                m = re.match(r"frame(\d+).png", image_filename)
                frame_id = int(m.group(1))
                colmap_pose_wxyz_xyz = np.array([float(x) for x in l1[1:8]])
                colmap_translate = colmap_pose_wxyz_xyz[-3:]
                colmap_euler = transformations.euler_from_quaternion(colmap_pose_wxyz_xyz[:4])
                colmap_pose_mat = transformations.compose_matrix(translate=colmap_translate, angles=colmap_euler)

                mocap_pose_mat = transforms_to_frame_id_map[frame_id]
                colmap_poses.append(colmap_pose_mat)
                mocap_poses.append(mocap_pose_mat)
                l2 = next(reader)  # skip  this line
            except StopIteration:
                break

    # for viz
    tf = TF2Wrapper()
    for _ in range(3):
        for i, (mocap_pose, colmap_pose) in enumerate(zip(mocap_poses, colmap_poses)):
            tf.send_transform_matrix(colmap_pose, parent='mocap_world', child=f'colmap_pose_{i}')
            tf.send_transform_matrix(mocap_pose, parent='mocap_world', child=f'mocap_pose_{i}')

    # solve for the transform between the poses, average over time
    mocap_to_colmaps = []
    for i, (mocap_pose, colmap_pose) in enumerate(zip(mocap_poses, colmap_poses)):
        mocap_to_colmap_i = np.linalg.solve(colmap_pose, mocap_pose)

        mocap_to_colmaps.append(mocap_to_colmap_i)

    # this is only correct if all the transforms are close
    avg_mocap_to_colmap = np.mean(mocap_to_colmaps, axis=0)
    avg_mocap_to_colmap_std = np.std(mocap_to_colmaps, axis=0)
    print(avg_mocap_to_colmap)
    print(avg_mocap_to_colmap_std)


if __name__ == '__main__':
    main()
