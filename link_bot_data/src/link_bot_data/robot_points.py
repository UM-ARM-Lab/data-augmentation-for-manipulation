import pathlib
import pickle

import pyjacobian_follower
import tensorflow as tf

from link_bot_pycommon.pycommon import unordered_pairs
from moonshine.tensorflow_utils import repeat_tensor
from moonshine.numpify import numpify


class RobotVoxelgridInfo:

    def __init__(self, joint_positions_key: str, robot_points_path=None):
        if robot_points_path is None:
            self.robot_points_path = pathlib.Path("robot_points_data/val/robot_points.pkl")
        else:
            self.robot_points_path = robot_points_path

        with self.robot_points_path.open("rb") as file:
            data = pickle.load(file)
        robot_points = data['points']
        self.res = data['res']
        self.link_names = list(robot_points.keys())
        self.points_per_links = []
        for link_name in self.link_names:
            self.points_per_links.append(len(robot_points[link_name]))
        self.n_points = sum(self.points_per_links)

        self.points_link_frame_list = get_points_link_frame(robot_points)
        points_link_frame_concat = tf.concat(self.points_link_frame_list, axis=0)
        ones = tf.ones([self.n_points, 1], tf.float32)
        link_points_link_frame_homo = tf.concat([points_link_frame_concat, ones], axis=1)
        link_points_link_frame_homo = tf.expand_dims(link_points_link_frame_homo, axis=-1)

        self.points_link_frame = link_points_link_frame_homo
        self.joint_positions_key = joint_positions_key

        self._allowed_collidable_names_and_points = None

    def precompute_allowed_collidable_pairs(self, ignored_collisions):
        allowed_collidable_names_and_points = []
        names_and_points = list(zip(self.link_names, self.points_link_frame_list))
        for (name1, link2_points_link_frame), (name2, link1_points_link_frame) in unordered_pairs(names_and_points):
            if {name1, name2} not in ignored_collisions:
                allowed_collidable_names_and_points.append(((name1, link2_points_link_frame),
                                                            (name2, link1_points_link_frame)))
        self._allowed_collidable_names_and_points = allowed_collidable_names_and_points

    def allowed_collidable_pairs(self):
        return self._allowed_collidable_names_and_points


def get_points_link_frame(points):
    points_link_frame = []
    for link_name, link_points_link_frame in points.items():
        link_points_link_frame = tf.cast(link_points_link_frame, dtype=tf.float32)
        points_link_frame.append(link_points_link_frame)
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


def batch_transform_robot_points(link_to_robot_transforms, robot_info: RobotVoxelgridInfo, batch_size):
    points_link_frame_homo_batch = repeat_tensor(robot_info.points_link_frame, batch_size, 0, True)
    links_to_robot_transform_batch = tf.repeat(link_to_robot_transforms, robot_info.points_per_links, axis=1)
    points_robot_frame_homo_batch = tf.matmul(links_to_robot_transform_batch, points_link_frame_homo_batch)
    points_robot_frame_batch = points_robot_frame_homo_batch[:, :, :3, 0]
    return points_robot_frame_batch
