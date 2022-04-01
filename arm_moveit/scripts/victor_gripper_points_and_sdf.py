#!/usr/bin/env python
import argparse
import pathlib
import pickle

import numpy as np
import pyrobot_points_generator
import tensorflow as tf
from tqdm import tqdm

import ros_numpy
import rospy
from arc_utilities import ros_init
from link_bot_pycommon.grid_utils_np import extent_res_to_origin_point, extent_to_env_shape
from moonshine.numpify import numpify
from moonshine.tfa_sdf import build_sdf_3d
from rviz_voxelgrid_visuals import conversions
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped
from sensor_msgs.msg import PointCloud2


def segment(pc, sdf, origin_point, res, threshold):
    """

    Args:
        pc: [n, 3], as set of n (x,y,z) points in the same frame as the voxel grid
        sdf: [h, w, c], signed distance field
        origin_point: [3], the (x,y,z) position of voxel [0,0,0]
        res: scalar, size of one voxel in meters
        threshold: the distance threshold determining what's segmented

    Returns:
        [m, 3] the segmented points

    """
    indices = batch_point_to_idx(pc, res, origin_point)
    in_bounds = tf.logical_not(tf.logical_or(tf.reduce_any(indices <= 0, -1), tf.reduce_any(indices >= sdf.shape, -1)))
    in_bounds_indices = tf.boolean_mask(indices, in_bounds, axis=0)
    in_bounds_pc = tf.boolean_mask(pc, in_bounds, axis=0)
    distances = tf.gather_nd(sdf, in_bounds_indices)
    close = distances < threshold
    segmented_points = tf.boolean_mask(in_bounds_pc, close, axis=0)
    return segmented_points


def visualize_pc(pub, points, frame_id='world'):
    list_of_tuples = [(p[0], p[1], p[2]) for p in points]
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
    np_record_array = np.array(list_of_tuples, dtype=dtype)
    msg = ros_numpy.msgify(PointCloud2, np_record_array, frame_id=frame_id, stamp=rospy.Time.now())
    pub.publish(msg)


def round_to_res(x, res):
    # helps with stupid numerics issues
    return tf.cast(tf.round(x / res), tf.int64)


def batch_point_to_idx(points, res, origin_point):
    """

    Args:
        points: [b,3] points in a frame, call it world
        res: [b] meters
        origin_point: [b,3] the position [x,y,z] of the center of the voxel (0,0,0) in the same frame as points

    Returns:

    """
    return round_to_res((points - origin_point), tf.expand_dims(res, -1))


def points_to_voxel_grid_res_origin_point(points, res, origin_point, h, w, c):
    """
    Args:
        points: [n, 3]
        res: [n]
        origin_point: [n, 3]
        h:
        w:
        c:

    Returns: 1-channel binary voxel grid
    """
    n = points.shape[0]
    indices = batch_point_to_idx(points, res, origin_point)  # [n, 3]
    ones = tf.ones([n])
    voxel_grid = tf.scatter_nd(indices, ones, [h, w, c])
    voxel_grid = tf.clip_by_value(voxel_grid, 0, 1)
    return voxel_grid


def get_grid_points(origin_point, res, shape):
    indices = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.stack(indices, axis=-1)
    indices = np.transpose(indices, [1, 0, 2, 3])
    points = (indices * res) + origin_point
    return points


def visualize_sdf(pub, sdf: np.ndarray, shape, res, origin_point, frame_id='world'):
    points = get_grid_points(origin_point, res, shape)
    list_of_tuples = [(p[0], p[1], p[2], d) for p, d in zip(points.reshape([-1, 3]), sdf.flatten())]
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('distance', np.float32)]
    np_record_array = np.array(list_of_tuples, dtype=dtype)
    msg = ros_numpy.msgify(PointCloud2, np_record_array, frame_id=frame_id, stamp=rospy.Time.now())
    pub.publish(msg)


@ros_init.with_ros("generate_robot_pointcloud")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=float, default=0.004)
    args = parser.parse_args()

    exclude_links = []

    outdir = pathlib.Path("robot_points_data/victor_gripper")
    outdir.mkdir(exist_ok=True, parents=True)
    outfilename = outdir / 'points_and_sdf.pkl'

    robot_points_generator = pyrobot_points_generator.RobotPointsGenerator(args.res, "robot_description",
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
    segmented_pub = rospy.Publisher("segmented", PointCloud2, queue_size=10, latch=True)
    test_pc_pub = rospy.Publisher("test", PointCloud2, queue_size=10, latch=True)
    pub = rospy.Publisher('vg', VoxelgridStamped, queue_size=1)

    visualize_sdf(sdf_pub, sdf.numpy(), shape, args.res, origin_point, frame_id='l_palm')
    pub.publish(conversions.vox_to_voxelgrid_stamped(vg.numpy(),
                                                     scale=args.res,
                                                     frame_id='l_palm',
                                                     origin=origin_point))

    data = {
        'robot_name':     robot_points_generator.get_robot_name(),
        'res':            args.res,
        'points':         points,
        'sdf':            sdf,
        'origin_point':   origin_point,
        'included_links': included_links,
    }

    with outfilename.open("wb") as outfile:
        pickle.dump(numpify(data), outfile)
    print(f"Wrote {outfilename.as_posix()}")

    z = 0.35
    test_points = tf.random.uniform([1_000, 3], [-z, -z, -z], [z, z, z])
    segmented_points = segment(test_points, sdf, origin_point, args.res, threshold=0.05)
    visualize_pc(test_pc_pub, test_points, frame_id='l_palm')
    visualize_pc(segmented_pub, segmented_points, frame_id='l_palm')


if __name__ == "__main__":
    main()
