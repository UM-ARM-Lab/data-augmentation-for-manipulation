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


def matvecmul(mat, vec):
    """
    Matrix-vector multiplication
    :param mat:
    :param vec:
    :return:
    """
    return tf.linalg.LinearOperatorFullMatrix(mat).matvec(vec)


def matmatmul(mat1, mat2):
    """
    Matrix-matrix multiplication
    :param mat1:
    :param mat2:
    :return:
    """
    return tf.linalg.LinearOperatorFullMatrix(mat1).matmul(mat2)


def angular_vel_tensor(w):
    if w.shape.ndims == 1:
        return tf.stack([[0., -w[2], w[1]],
                         [w[2], 0., -w[0]],
                         [-w[1], w[0], 0.]])
    else:
        di = tf.zeros_like(w[:, 0])
        return tf.transpose(
            tf.stack([[di, -w[:, 2], w[:, 1]],
                      [w[:, 2], di, -w[:, 0]],
                      [-w[:, 1], w[:, 0], di]]),
            perm=(2, 0, 1)
        )
