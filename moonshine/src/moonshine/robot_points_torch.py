import pathlib
import pickle

import numpy as np
import pyjacobian_follower
import torch

from link_bot_pycommon.pycommon import unordered_pairs
from moonshine.numpify import numpify
from moonshine.torch_utils import repeat_tensor


class RobotVoxelgridInfo:

    def __init__(self, joint_positions_key: str, robot_points_path=None, exclude_links=None):
        if robot_points_path is None:
            self.robot_points_path = pathlib.Path("robot_points_data/val/robot_points.pkl")
        else:
            self.robot_points_path = robot_points_path
        if exclude_links is None:
            self.exclude_links = []
        else:
            self.exclude_links = exclude_links

        with self.robot_points_path.open("rb") as file:
            data = pickle.load(file)

        robot_points = data['points']
        self.res = data['res']
        self.link_names = list(robot_points.keys())
        for k in self.exclude_links:
            self.link_names.remove(k)

        self.points_per_links = []
        for link_name in self.link_names:
            self.points_per_links.append(len(robot_points[link_name]))
        self.n_points = sum(self.points_per_links)

        self.points_link_frame_list = get_points_link_frame(robot_points, self.exclude_links)
        points_link_frame_concat = torch.cat(self.points_link_frame_list, 0)
        ones = torch.ones(self.n_points, 1, dtype=torch.float32)
        link_points_link_frame_homo = torch.cat([points_link_frame_concat, ones], 1)
        link_points_link_frame_homo = link_points_link_frame_homo.unsqueeze(-1)

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


def get_points_link_frame(points, exclude_links):
    points_link_frame = []
    for link_name, link_points_link_frame in points.items():
        if link_name in exclude_links:
            continue
        link_points_link_frame = torch.from_numpy(np.array(link_points_link_frame, dtype=np.float32))
        points_link_frame.append(link_points_link_frame)
    return points_link_frame


def batch_robot_state_to_transforms(jacobian_follower: pyjacobian_follower.JacobianFollower,
                                    names,
                                    positions,
                                    link_names):
    """ returns the transforms from the robot base link (according to moveit) to each link """
    link_to_robot_transforms = jacobian_follower.batch_get_link_to_robot_transforms(names,
                                                                                    numpify(positions),
                                                                                    link_names)
    link_to_robot_transforms = torch.from_numpy(np.array(link_to_robot_transforms, dtype=np.float32))
    return link_to_robot_transforms  # [b, n_links, 4, 4]


def batch_transform_robot_points(link_to_robot_transforms, robot_info: RobotVoxelgridInfo, batch_size):
    points_link_frame_homo_batch = repeat_tensor(robot_info.points_link_frame, batch_size, 0, True)  # [b, points, 4, 1]
    links_to_robot_transform_batch = link_to_robot_transforms.repeat_interleave(
        torch.tensor(robot_info.points_per_links), dim=1)  # [b, points, 4, 4]
    points_robot_frame_homo_batch = torch.matmul(links_to_robot_transform_batch,
                                                 points_link_frame_homo_batch)  # [b, points, 4, 1]
    points_robot_frame_batch = points_robot_frame_homo_batch[:, :, :3, 0]  # [b, points, 3]
    return points_robot_frame_batch
