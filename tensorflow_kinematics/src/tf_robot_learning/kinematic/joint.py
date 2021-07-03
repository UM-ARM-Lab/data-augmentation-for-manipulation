# tf_robot_learning, a all-around tensorflow library for robotics.
#
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

from enum import IntEnum

import tensorflow as tf

from tf_robot_learning.kinematic.frame import Frame, Twist
from tf_robot_learning.kinematic.rotation import rot_2, rot_x, rot_y, rot_z, twist_x, twist_y, twist_z, twist_2


class JointType(IntEnum):
    RotX = 0
    RotY = 1
    RotZ = 2
    RotAxis = 3
    NoneT = 4


class Joint:
    def __init__(self, joint_type, origin=None, axis=None, name='', limits=None):
        self.type = joint_type
        self.name = name

        if limits is not None:
            self.limits = {'up':  limits.upper, 'low': limits.lower,
                           'vel': limits.velocity, 'effort': limits.effort}
        else:
            self.limits = {}

        self.axis = axis
        self.origin = origin

        self.pose_0 = self.pose(tf.zeros([1], tf.float32), 1)

    def pose(self, a, batch_size: int):
        if self.type is JointType.RotX:
            return Frame(m=rot_x(a), batch_shape=batch_size)
        elif self.type is JointType.RotY:
            return Frame(m=rot_y(a), batch_shape=batch_size)
        elif self.type is JointType.RotZ:
            return Frame(m=rot_z(a), batch_shape=batch_size)
        elif self.type is JointType.RotAxis:
            return Frame(p=self.origin, m=rot_2(self.axis, a), batch_shape=batch_size)
        elif self.type is JointType.NoneT:
            return Frame(batch_shape=batch_size)
        else:
            raise NotImplementedError()

    def twist(self, a):
        if self.type is JointType.RotX:
            return Twist(twist_x(a))
        elif self.type is JointType.RotY:
            return Twist(twist_y(a))
        elif self.type is JointType.RotZ:
            return Twist(twist_z(a))
        elif self.type is JointType.RotAxis:
            return Twist(twist_2(self.axis, a))
        elif self.type is JointType.NoneT:
            return Twist()


class Link:
    def __init__(self, frame):
        self.frame = frame

    def pose(self):
        return self.frame


SUPPORTED_JOINT_TYPES = ['revolute', 'fixed', 'prismatic']
SUPPORTED_ACTUATED_JOINT_TYPES = ['revolute', 'prismatic']
