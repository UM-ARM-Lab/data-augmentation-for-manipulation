# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Emmanuel Pignat <emmanuel.pignat@idiap.ch>,
#
# This file is part of tf_robot_learning.
#
# tf_robot_learning is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# tf_robot_learning is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tf_robot_learning. If not, see <http://www.gnu.org/licenses/>.
from typing import List, Dict, Tuple

import tensorflow as tf

from tf_robot_learning.kinematic.frame import Frame
from tf_robot_learning.kinematic.joint import JointType, Joint, Link
from tf_robot_learning.kinematic.segment import Segment


class Tree:
    def __init__(self, urdf, segments_map: Dict[str, List[Tuple[Joint, Link]]]):
        self.urdf = urdf
        self.segments_map = segments_map
        self.nb_segm = len(segments_map)

        self._joint_limits = None
        self._names = None
        self._nb_joints = None

    def _fk(self, frames: Dict[str, Frame], q, parent_link_name: str, parent_frame: Frame, batch_size: int):
        joint: Joint
        segment: Segment
        for (joint, segment) in self.segments_map[parent_link_name]:
            if segment.joint.type is not JointType.NoneT:
                j = self.actuated_joint_names().index(segment.joint.name)
                link_frame = parent_frame * segment.pose(q[:, j], batch_size)
            else:
                link_frame = parent_frame * segment.pose(tf.zeros([1], dtype=tf.float32), 1)
            frames[segment.child_name] = link_frame

            if segment.child_name in self.segments_map:
                self._fk(frames, q, segment.child_name, link_frame, batch_size)

    def fk(self, q):
        """
        Pose of all segments of the tree

        :param q:		[batch_size, nb_joint] or [nb_joint] or list of [batch_size] Joint angles
        :return:
        output order is based on order of segments, which is constructed via DFS over the URDF
        """
        assert q.shape[1] == self.get_num_joints()

        batch_size = q.shape[0]
        root_frame = Frame(batch_shape=batch_size)
        root_name = self.urdf.get_root()
        frames = {
            root_name: root_frame,  # could use this to represent transform from robot to world
        }
        self._fk(frames, q, root_name, root_frame, batch_size)

        poses = {k: frame.xm for k, frame in frames.items()}
        return poses

    def get_joint_limits(self):
        if self._joint_limits is None:
            limits = [[j.limit.lower, j.limit.upper] for j in self.urdf.joint_map.values() if j.type != 'fixed']
            self._joint_limits = tf.constant(limits, dtype=tf.float32)

        return self._joint_limits

    def actuated_joint_names(self):
        if self._names is None or self._names is None:
            self._names = [name for name, j in self.urdf.joint_map.items() if j.type != 'fixed']
        return self._names

    def get_num_joints(self):
        if self._nb_joints is None:
            self._nb_joints = len(self.actuated_joint_names())
        return self._nb_joints
