# tensorflow_kinematics, a all-around tensorflow library for robotics.
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Emmanuel Pignat <emmanuel.pignat@idiap.ch>,
#
# This file is part of tensorflow_kinematics.
#
# tensorflow_kinematics is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# tensorflow_kinematics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tensorflow_kinematics. If not, see <http://www.gnu.org/licenses/>.
from tensorflow_kinematics.tf_utils import matvecmul
import tensorflow as tf


class Segment:
    def __init__(self, joint, f_tip, child_name='', link=None):
        """
        Segment of a kinematic chain

        :param joint:
        :type joint: tk.Joint
        :param f_tip:
        :type f_tip: tk.Frame
        """
        self.joint = joint
        self.f_tip = joint.pose(tf.zeros([1]), 1).inv() * f_tip

        self.child_name = child_name

        self.link = link

        self.pose_0 = self.pose(tf.zeros([1], tf.float32), 1)

    def pose(self, q, batch_size):
        return self.joint.pose(q, batch_size) * self.f_tip

    def twist(self, q, qdot=0.):
        return self.joint.twist(qdot).ref_point(matvecmul(self.joint.pose(q).m, self.f_tip.p))
