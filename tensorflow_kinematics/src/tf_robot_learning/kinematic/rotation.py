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

import tensorflow as tf


def twist_x(a):
    return tf.stack([0., 0., 0., a, 0., 0.])


def twist_y(a):
    return tf.stack([0., 0., 0., 0., a, 0.])


def twist_z(a):
    return tf.stack([0., 0., 0., 0., 0., a])


def twist_2(axis, a):
    if isinstance(a, float) or a.shape.ndims == 0:
        return tf.concat([tf.zeros(3), axis * a], axis=0)
    else:
        return tf.concat([tf.zeros(3)[None] * a, axis[None] * a], axis=1)


def skew_x(u):
    zeros = tf.zeros([u.shape[0]])
    return tf.transpose(tf.stack([[zeros, -u[:, 2], u[:, 1]],
                                  [u[:, 2], zeros, -u[:, 0]],
                                  [-u[:, 1], u[:, 0], zeros]]), [2, 0, 1])


def rot_2(axis, a):
    """
    https://en.wikipedia.org/wiki/Rotation_matrix see Rotation matrix from axis and angle
    :param axis:
    :param a:
    :return:
    """
    if isinstance(a, float) or a.shape.ndims == 0:
        return tf.cos(a) * tf.eye(3) + tf.sin(a) * skew_x(axis[None])[0] + (1 - tf.cos(a)) * tf.einsum('i,j->ij', axis,
                                                                                                       axis)
    else:
        return tf.cos(a)[:, None, None] * tf.eye(3)[None] + tf.sin(a)[:, None, None] * skew_x(axis) + \
               (1 - tf.cos(a)[:, None, None]) * tf.einsum('bi,bj->bij', axis, axis)


def rpy(rpy):
    """
    http://planning.cs.uiuc.edu/node102.html
    :param rpy:
    :return:
    """
    m_r = rot_x(-rpy[..., 0])
    m_p = rot_y(-rpy[..., 1])
    m_y = rot_z(-rpy[..., 2])

    return tf.matmul(m_y, tf.matmul(m_p, m_r))


def rot_x(a):
    cs = tf.cos(a)
    sn = tf.sin(a)
    if a.shape.ndims == 1:
        _ones, _zeros = tf.ones_like(cs), tf.zeros_like(cs)
        return tf.transpose(tf.stack([[_ones, _zeros, _zeros],
                                      [_zeros, cs, sn],
                                      [_zeros, -sn, cs]]), (2, 0, 1))
    else:

        return tf.stack([[1., 0., 0.],
                         [0., cs, sn],
                         [0., -sn, cs]])


def rot_y(a):
    cs = tf.cos(a)
    sn = tf.sin(a)

    if a.shape.ndims == 1:
        _ones, _zeros = tf.ones_like(cs), tf.zeros_like(cs)

        return tf.transpose(tf.stack([[cs, _zeros, -sn],
                                      [_zeros, _ones, _zeros],
                                      [sn, _zeros, cs]]), (2, 0, 1))

    else:

        return tf.stack([[cs, 0., -sn],
                         [0., 1., 0.],
                         [sn, 0., cs]])


def rot_z(a):
    cs = tf.cos(a)
    sn = tf.sin(a)
    if a.shape.ndims == 1:
        _ones, _zeros = tf.ones_like(cs), tf.zeros_like(cs)

        return tf.transpose(tf.stack([[cs, sn, _zeros],
                                      [-sn, cs, _zeros],
                                      [_zeros, _zeros, _ones]]), (2, 0, 1))

    else:

        return tf.stack([[cs, sn, 0.],
                         [-sn, cs, 0.],
                         [0., 0., 1.]])
