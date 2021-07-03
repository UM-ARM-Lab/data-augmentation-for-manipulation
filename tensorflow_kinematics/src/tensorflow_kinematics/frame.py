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

import tensorflow as tf

from tensorflow_kinematics.layout import FkLayout
from tensorflow_kinematics.tf_utils import angular_vel_tensor, matmatmul, matvecmul


class Rotation(tf.Tensor):
    pass

    @property
    def is_batch(self):
        return self.shape.ndims == 3


class Twist:
    def __init__(self, dx=tf.zeros(6)):
        self.dx = dx

    @property
    def is_batch(self):
        return self.dx.shape.ndims == 2

    def dx_mat(self, m, layout=FkLayout.xm):
        """
        https://en.wikipedia.org/wiki/Angular_velocity
        :param m:
        :return:
        """
        w = angular_vel_tensor(self.rot)
        dm_dphi = matmatmul(w, m)

        if layout is FkLayout.xmv:
            if self.is_batch:
                return tf.concat(
                    [self.vel, tf.reshape(tf.transpose(dm_dphi, perm=(0, 2, 1)), (-1, 9))], axis=1
                )
            else:
                return tf.concat(
                    [self.vel, tf.reshape(tf.transpose(dm_dphi, perm=(1, 0)), (9,))], axis=0
                )

        else:
            if self.is_batch:
                return tf.concat(
                    [self.vel, tf.reshape(dm_dphi, (-1, 9))], axis=1
                )
            else:
                return tf.concat(
                    [self.vel, tf.reshape(dm_dphi, (9,))], axis=0
                )

    @property
    def vel(self):
        if self.is_batch:
            return self.dx[:, :3]
        else:
            return self.dx[:3]

    @property
    def rot(self):
        if self.is_batch:
            return self.dx[:, 3:]
        else:
            return self.dx[3:]

    def ref_point(self, v):
        if v.shape.ndims > self.rot.shape.ndims:
            rot = self.rot[None] * (tf.zeros_like(v) + 1.)
            vel = self.vel + tf.linalg.cross(rot, v)
        elif v.shape.ndims < self.rot.shape.ndims:
            n = self.rot.shape[0].value
            vel = self.vel + tf.linalg.cross(self.rot, v[None] * tf.ones((n, 1)))
            rot = self.rot
        else:
            vel = self.vel + tf.linalg.cross(self.rot, v)
            rot = self.rot

        if self.is_batch or v.shape.ndims == 2:
            return Twist(tf.concat([vel, rot], 1))
        else:
            return Twist(tf.concat([vel, rot], 0))

    def __rmul__(self, other):
        if isinstance(other, Frame):
            raise NotImplementedError

        else:
            rot = matvecmul(other, self.rot)
            vel = matvecmul(other, self.vel)
            if self.is_batch or rot.shape.ndims == 2:
                return Twist(tf.concat([vel, rot], 1))
            else:
                return Twist(tf.concat([vel, rot], 0))


class Frame:
    def __init__(self, p=None, m=None, batch_shape=None):
        """
        :param p:
            Translation vector
        :param m:
            Rotation matrix
        :param batch_shape		int
        """

        if batch_shape is None:
            p = tf.zeros(3) if p is None else p
            m = tf.eye(3) if m is None else m
        else:
            p = tf.zeros((batch_shape, 3)) if p is None else p
            m = tf.eye(3, batch_shape=(batch_shape,)) if m is None else m
            assert m.shape[0] == batch_shape

        if isinstance(m, tf.Variable):
            _m = tf.identity(m)
        else:
            _m = m

        self.p = p
        self.m = _m

    @property
    def is_batch(self):
        return self.m.shape.ndims == 3

    @property
    def xm2(self):
        """
        4x4 transformation matrix
        (order : 'C' - last index changing the first)
        :return:
        """
        assert self.is_batch
        b = self.m.shape[0]
        m34 = tf.concat([self.p[:, :, None], self.m], axis=-1)
        m44 = tf.concat([m34, tf.constant([[[0, 0, 0, 1]]]*b, dtype=tf.float32)], axis=-2)
        return m44

    @property
    def xm(self):
        """
        Position and vectorized rotation matrix
        (order : 'C' - last index changing the first)
        :return:
        """
        if self.is_batch:
            return tf.concat([self.p, tf.reshape(self.m, [-1, 9])], axis=1)
        else:
            return tf.concat([self.p, tf.reshape(self.m, [9])], axis=0)

    @property
    def xmv(self):
        """
        Position and vectorized rotation matrix
        (order : 'C' - last index changing the first)
        :return:
        """
        if self.is_batch:
            return tf.concat([self.p, tf.reshape(tf.transpose(self.m, perm=(0, 2, 1)), [-1, 9])], axis=1)
        else:
            return tf.concat([self.p, tf.reshape(tf.transpose(self.m, perm=(1, 0)), [9])], axis=0)

    @property
    def xq(self):
        """
        Position and Quaternion
        :return:
        """
        raise NotImplementedError()

    def inv(self):
        return Frame(p=-tf.squeeze(tf.matmul(self.m, tf.expand_dims(self.p, -1), transpose_a=True), -1),
                     m=tf.transpose(self.m, [0, 2, 1]))

    def __mul__(self, other):
        if isinstance(other, Twist):
            # TODO check
            rot = matvecmul(self.m, other.rot)
            vel = matvecmul(self.m, other.vel) + tf.linalg.cross(self.p, rot)  # should be cross product
            return Twist(tf.concat([vel, rot], 0))

        elif isinstance(other, tf.Tensor) or isinstance(other, tf.Variable):
            if other.shape[-1].value == 3:  # only position
                if self.is_batch:
                    # return tf.einsum('aij,bj->abi', self.m, other) + self.p[:, None]
                    return tf.linalg.LinearOperatorFullMatrix(self.m[:, None]).matvec(other[None]) + self.p[None]
                else:
                    return tf.linalg.LinearOperatorFullMatrix(self.m).matvec(other) + self.p[None]
            else:
                raise NotImplementedError('Only position supported yet')
        else:
            return Frame(p=matvecmul(self.m, other.p) + self.p,
                         m=matmatmul(self.m, other.m))
