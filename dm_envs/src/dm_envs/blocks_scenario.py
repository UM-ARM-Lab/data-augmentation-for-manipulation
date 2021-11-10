import re
from typing import Dict

import tensorflow as tf
from matplotlib import colors

from dm_envs.blocks_task import PlanarPushingBlocksTask
from dm_envs.planar_pushing_scenario import PlanarPushingScenario, transformation_matrices_from_pos_quat
from moonshine.geometry import transform_points_3d
from moonshine.moonshine_utils import repeat_tensor
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker


def blocks_to_points(positions, quats, size):
    """

    Args:
        positions:  [b, m, T, 3]
        quats:  [b, m, T, 4]
        size:  [b, T]

    Returns: [b, m, T, 8, 3]

    """
    m = positions.shape[1]
    sized_cube_points = size_to_points(size)  # [b, T, 8, 3]
    sized_cubes_points = repeat_tensor(sized_cube_points, m, axis=1, new_axis=True)  # [b, m, T, 8, 3]
    transform_matrix = transformation_matrices_from_pos_quat(positions, quats)  # [b, m, T, 4, 4]
    transform_matrix = repeat_tensor(transform_matrix, 8, 3, True)
    obj_points = transform_points_3d(transform_matrix, sized_cubes_points)  # [b, m, T, 8, 3]
    return obj_points


def size_to_points(size):
    """

    Args:
        size: [b, T] where each element is the size of the cube. it can vary of batch/time

    Returns: [b, T, 8, 3] where 8 represents the corners the cube, and 3 is x,y,z.

    """
    unit_cube_points = tf.constant([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                                    [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]], tf.float32) - 0.5
    return unit_cube_points[None, None] * size[:, :, None, None]  # [b, T, 8, 3]


def wxyz2xyzw(quat):
    w, x, y, z = tf.unstack(quat, axis=-1)
    return tf.stack([x, y, z, w], axis=-1)


class BlocksScenario(PlanarPushingScenario):

    def plot_state_rviz(self, state: Dict, **kwargs):
        super().plot_state_rviz(state, **kwargs)

        ns = kwargs.get("label", "")
        color_msg = ColorRGBA(*colors.to_rgba(kwargs.get("color", "r")))
        if 'a' in kwargs:
            color_msg.a = kwargs['a']
            a = kwargs['a']
        else:
            a = 1.0
        idx = kwargs.get("idx", 0)

        num_objs = state['num_objs'][0]
        block_size = state['block_size'][0]
        msg = MarkerArray()
        for i in range(num_objs):
            block_position = state[f'obj{i}/position']
            block_orientation = state[f'obj{i}/orientation']

            block_marker = Marker()
            block_marker.header.frame_id = 'world'
            block_marker.action = Marker.ADD
            block_marker.type = Marker.CUBE
            block_marker.id = idx * num_objs + i
            block_marker.ns = ns
            block_marker.color = color_msg
            block_marker.pose.position.x = block_position[0, 0]
            block_marker.pose.position.y = block_position[0, 1]
            block_marker.pose.position.z = block_position[0, 2]
            block_marker.pose.orientation.w = block_orientation[0, 0]
            block_marker.pose.orientation.x = block_orientation[0, 1]
            block_marker.pose.orientation.y = block_orientation[0, 2]
            block_marker.pose.orientation.z = block_orientation[0, 3]
            block_marker.scale.x = block_size
            block_marker.scale.y = block_size
            block_marker.scale.z = block_size
            msg.markers.append(block_marker)

        self.state_viz_pub.publish(msg)

    def compute_obj_points(self, inputs: Dict, num_object_interp: int, batch_size: int):
        """

        Args:
            inputs: contains the poses and size of the blocks, over a whole trajectory, which we convert into points
            num_object_interp:
            batch_size:

        Returns:

        """
        size = inputs['block_size'][:, :, 0]  # [b, T]
        num_objs = inputs['num_objs'][0, 0, 0]  # assumed fixed across batch/time
        positions = []  # [b, m, T, 3]
        quats = []  # [b, m, T, 4]
        for block_idx in range(num_objs):
            pos = inputs[f"obj{block_idx}/position"][:, :, 0]  # [b, T, 3]
            # in our mujoco dataset the quaternions are stored w,x,y,z but the rest of our code assumes xyzw
            quat = inputs[f"obj{block_idx}/orientation"][:, :, 0]  # [b, T, 4]
            quat = wxyz2xyzw(quat)
            positions.append(pos)
            quats.append(quat)
        positions = tf.stack(positions, axis=1)
        quats = tf.stack(quats, axis=1)

        obj_points = blocks_to_points(positions, quats, size)

        # combine to get one set of points per object
        obj_points = tf.reshape(obj_points, [obj_points.shape[0], obj_points.shape[1], -1, 3])

        return obj_points

    @staticmethod
    def is_points_key(k):
        return any([
            re.match('obj.*position', k),
            k == 'jaco_arm/primitive_hand/tcp_pos',
        ])

    def make_dm_task(self, params):
        return PlanarPushingBlocksTask(params)

    def __repr__(self):
        return "blocks"
