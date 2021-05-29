import pathlib
import pickle

import pyjacobian_follower
import tensorflow as tf

from moonshine.moonshine_utils import numpify, repeat_tensor


class RobotVoxelgridInfo:

    def __init__(self):
        self.robot_points_filename = pathlib.Path("robot_points_data/val/robot_points.pkl")
        with self.robot_points_filename.open("rb") as file:
            data = pickle.load(file)
        robot_points = data['points']
        self.link_names = list(robot_points.keys())
        self.points_per_links, self.points_link_frame = setup_robot_points(points=robot_points,
                                                                           link_names=self.link_names)


def get_points_link_frame(points):
    points_link_frame = []
    for link_name, link_points_link_frame in points.items():
        link_points_link_frame = tf.cast(link_points_link_frame, dtype=tf.float32)
        points_link_frame.append(link_points_link_frame)
    points_link_frame = tf.concat(points_link_frame, axis=0)
    return points_link_frame


def batch_robot_state_to_transforms(jacobian_follower: pyjacobian_follower.JacobianFollower,
                                    names,
                                    positions,
                                    link_names):
    link_to_robot_transforms = jacobian_follower.batch_get_link_to_robot_transforms(names,
                                                                                    numpify(positions),
                                                                                    link_names)
    link_to_robot_transforms = tf.cast(link_to_robot_transforms, tf.float32)
    return link_to_robot_transforms


def batch_transform_robot_points(jacobian_follower: pyjacobian_follower.JacobianFollower,
                                 names,
                                 positions,
                                 robot_info: RobotVoxelgridInfo,
                                 batch_size):
    points_link_frame_homo_batch = repeat_tensor(robot_info.points_link_frame, batch_size, 0, True)
    link_to_robot_transforms = batch_robot_state_to_transforms(jacobian_follower, names, positions,
                                                               robot_info.link_names)
    links_to_robot_transform_batch = tf.repeat(link_to_robot_transforms, robot_info.points_per_links, axis=1)
    points_robot_frame_homo_batch = tf.matmul(links_to_robot_transform_batch, points_link_frame_homo_batch)
    points_robot_frame_batch = points_robot_frame_homo_batch[:, :, :3, 0]
    return points_robot_frame_batch


def setup_robot_points(points, link_names):
    points_per_links = []
    for link_name in link_names:
        points_per_links.append(len(points[link_name]))
    n_points = sum(points_per_links)
    points_link_frame = get_points_link_frame(points)
    ones = tf.ones([n_points, 1], tf.float32)
    link_points_link_frame_homo = tf.concat([points_link_frame, ones], axis=1)
    link_points_link_frame_homo = tf.expand_dims(link_points_link_frame_homo, axis=-1)
    return points_per_links, link_points_link_frame_homo
