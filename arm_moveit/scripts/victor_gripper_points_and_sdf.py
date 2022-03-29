#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import argparse
import pathlib
import pickle

import pyrobot_points_generator
from tqdm import tqdm

import rospy
from arc_utilities import ros_init
from link_bot_pycommon.grid_utils_np import extent_res_to_origin_point, extent_to_env_shape
from moonshine.raster_3d_tf import points_to_voxel_grid_res_origin_point
from moonshine.tfa_sdf import build_sdf_3d, visualize_sdf
from rviz_voxelgrid_visuals import conversions
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped
from sensor_msgs.msg import PointCloud2


@ros_init.with_ros("generate_robot_pointcloud")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=float, default=0.004)
    args = parser.parse_args()

    exclude_links = []

    outdir = pathlib.Path("robot_points_data/victor_gripper")
    outdir.mkdir(exist_ok=True, parents=True)
    outfilename = outdir / 'points_and_sdf.pkl'
    if outfilename.exists():
        q = input(f"File {outfilename.as_posix()} already exist, do you want to overwrite? [Y/n]")
        if q == 'n':
            return

    robot_points_generator = pyrobot_points_generator.RobotPointsGenerator(args.res, "victor/robot_description",
                                                                           args.res / 2)
    links = robot_points_generator.get_link_names()
    points = []
    included_links = []
    for link_to_check in tqdm(links):
        if link_to_check in exclude_links:
            continue
        if 'l_' != link_to_check[:2]:
            continue

        p = robot_points_generator.check_collision(link_to_check, 'l_palm')
        if len(p) > 0:
            included_links.append(link_to_check)
            points.extend(p)

    points = tf.convert_to_tensor(points, dtype=tf.float32)
    padding = 0.1
    lower = tf.reduce_min(points, 0) - padding
    upper = tf.reduce_max(points, 0) + padding
    extent = tf.reshape(tf.stack([lower, upper], 1), -1)
    origin_point = extent_res_to_origin_point(extent, args.res)
    shape = extent_to_env_shape(extent, args.res)
    h, w, c = shape
    vg = points_to_voxel_grid_res_origin_point(points, args.res, origin_point, h, w, c)
    sdf = build_sdf_3d(vg[None], tf.convert_to_tensor([args.res]))[0]

    # viusualize
    sdf_pub = rospy.Publisher("sdf", PointCloud2, queue_size=10, latch=True)
    sdf_viz = tf.transpose(sdf, [1, 0, 2]).numpy()
    visualize_sdf(sdf_pub, sdf_viz, shape, args.res, origin_point, frame_id='robot_root')
    pub = rospy.Publisher('vg', VoxelgridStamped, queue_size=1)
    pub.publish(conversions.vox_to_voxelgrid_stamped(vg.numpy(),
                                                     scale=args.res,
                                                     frame_id='robot_root',
                                                     origin=[0, 0, 0]))

    data = {
        'robot_name': robot_points_generator.get_robot_name(),
        'res':        args.res,
        'points':     points,
        'sdf':        sdf.numpy(),
    }

    with outfilename.open("wb") as outfile:
        pickle.dump(data, outfile)
    print(f"Wrote {outfilename.as_posix()}")


if __name__ == "__main__":
    main()
