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

from enum import IntEnum


class FkLayout(IntEnum):
    x = 0  # only position
    xq = 1  # position - quaternion
    xm = 2  # position - vectorized rotation matrix - order 'C'
    xmv = 4  # position - vectorized rotation matrix - order 'F' - [x_1, x_2, x_3, y_1, ..., z_3]
    f = 3  # frame.
