import re
from typing import Dict

import tensorflow as tf
from matplotlib import colors
from pyjacobian_follower import IkParams
from tensorflow_graphics.geometry.transformation import quaternion

from dm_envs.cylinders_env import PlanarPushingCylindersTask
from dm_envs.planar_pushing_scenario import PlanarPushingScenario, transformation_matrices_from_pos_quat
from link_bot_data.dataset_utils import add_predicted
from moonshine.geometry import transform_points_3d
from moonshine.moonshine_utils import repeat_tensor
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker


def cylinders_to_points(positions, quats, height, radius):
    """

    Args:
        positions:  [b, m, T, 3]
        quats:  [b, m, T, 4]
        height:  [b, T]
        radius:  [b, T]

    Returns: [b, m, T, 8, 3]

    """
    raise NotImplementedError()
    m = positions.shape[1]
    sized_points = size_to_points(height, radius)  # [b, T, 8, 3]
    sized_points = repeat_tensor(sized_points, m, axis=1, new_axis=True)  # [b, m, T, 8, 3]
    transform_matrix = transformation_matrices_from_pos_quat(positions, quats)  # [b, m, T, 4, 4]
    transform_matrix = repeat_tensor(transform_matrix, 8, 3, True)
    obj_points = transform_points_3d(transform_matrix, sized_points)  # [b, m, T, 8, 3]
    return obj_points


def size_to_points(height, radius):
    """

    Args:
        height: [b, T]
        radius: [b, T]

    Returns: [b, T, 8, 3] where 8 represents the corners the cube, and 3 is x,y,z.

    """
    unit_cylinder_points = tf.constant([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                                        [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]], tf.float32) - 0.5
    return unit_cylinder_points[None, None] * size[:, :, None, None]  # [b, T, 8, 3]


def wxyz2xyzw(quat):
    w, x, y, z = tf.unstack(quat, axis=-1)
    return tf.stack([x, y, z, w], axis=-1)


class CylindersScenario(PlanarPushingScenario):

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
        height = state['height'][0]
        radius = state['radius'][0]
        msg = MarkerArray()
        for i in range(num_objs):
            position = state[f'obj{i}/position']
            orientation = state[f'obj{i}/orientation']

            marker = Marker()
            marker.header.frame_id = 'world'
            marker.action = Marker.ADD
            marker.type = Marker.CYLINDER
            marker.id = idx * num_objs + i
            marker.ns = ns
            marker.color = color_msg
            marker.pose.position.x = position[0, 0]
            marker.pose.position.y = position[0, 1]
            marker.pose.position.z = position[0, 2]
            marker.pose.orientation.w = orientation[0, 0]
            marker.pose.orientation.x = orientation[0, 1]
            marker.pose.orientation.y = orientation[0, 2]
            marker.pose.orientation.z = orientation[0, 3]
            marker.scale.x = radius * 2
            marker.scale.y = radius * 2
            marker.scale.z = height
            msg.markers.append(marker)

        self.state_viz_pub.publish(msg)

    def compute_obj_points(self, inputs: Dict, num_object_interp: int, batch_size: int):
        """

        Args:
            inputs: contains the poses and size of the blocks, over a whole trajectory, which we convert into points
            num_object_interp:
            batch_size:

        Returns:

        """
        height = inputs['height'][:, :, 0]  # [b, T]
        radius = inputs['radius'][:, :, 0]  # [b, T]
        num_objs = inputs['num_objs'][0, 0, 0]  # assumed fixed across batch/time
        positions = []  # [b, m, T, 3]
        quats = []  # [b, m, T, 4]
        for i in range(num_objs):
            pos = inputs[f"obj{i}/position"][:, :, 0]  # [b, T, 3]
            # in our mujoco dataset the quaternions are stored w,x,y,z but the rest of our code assumes xyzw
            quat = inputs[f"obj{i}/orientation"][:, :, 0]  # [b, T, 4]
            quat = wxyz2xyzw(quat)
            positions.append(pos)
            quats.append(quat)
        positions = tf.stack(positions, axis=1)
        quats = tf.stack(quats, axis=1)

        obj_points = cylinders_to_points(positions, quats, height, radius)

        # combine to get one set of points per object
        obj_points = tf.reshape(obj_points, [obj_points.shape[0], obj_points.shape[1], -1, 3])

        return obj_points

    def apply_object_augmentation_no_ik(self,
                                        m,
                                        to_local_frame,
                                        inputs: Dict,
                                        batch_size,
                                        time,
                                        h: int,
                                        w: int,
                                        c: int,
                                        ):
        """

        Args:
            m: [b, 4, 4]
            to_local_frame: [b, 1, 3]  the 1 can also be equal to time
            inputs:
            batch_size:
            time:
            h:
            w:
            c:

        Returns:

        """
        # apply those to the rope and grippers
        rope_points = tf.reshape(inputs[add_predicted('rope')], [batch_size, time, -1, 3])
        left_gripper_point = inputs[add_predicted('left_gripper')]
        right_gripper_point = inputs[add_predicted('right_gripper')]
        left_gripper_points = tf.expand_dims(left_gripper_point, axis=-2)
        right_gripper_points = tf.expand_dims(right_gripper_point, axis=-2)

        def _transform(m, points, _to_local_frame):
            points_local_frame = points - _to_local_frame
            points_local_frame_aug = transform_points_3d(m, points_local_frame)
            return points_local_frame_aug + _to_local_frame

        # m is expanded to broadcast across batch & num_points dimensions
        rope_points_aug = _transform(m[:, None, None], rope_points, to_local_frame[:, None])
        left_gripper_points_aug = _transform(m[:, None, None], left_gripper_points, to_local_frame[:, None])
        right_gripper_points_aug = _transform(m[:, None, None], right_gripper_points, to_local_frame[:, None])

        # compute the new action
        left_gripper_position = inputs['left_gripper_position']
        right_gripper_position = inputs['right_gripper_position']
        # m is expanded to broadcast across batch dimensions
        left_gripper_position_aug = _transform(m[:, None], left_gripper_position, to_local_frame)
        right_gripper_position_aug = _transform(m[:, None], right_gripper_position, to_local_frame)

        rope_aug = tf.reshape(rope_points_aug, [batch_size, time, -1])
        left_gripper_aug = tf.reshape(left_gripper_points_aug, [batch_size, time, -1])
        right_gripper_aug = tf.reshape(right_gripper_points_aug, [batch_size, time, -1])

        # Now that we've updated the state/action in inputs, compute the local origin point
        state_aug_0 = {
            'left_gripper':  left_gripper_aug[:, 0],
            'right_gripper': right_gripper_aug[:, 0],
            'rope':          rope_aug[:, 0]
        }
        local_center_aug = self.local_environment_center_differentiable(state_aug_0)
        res = inputs['res']
        local_origin_point_aug = batch_center_res_shape_to_origin_point(local_center_aug, res, h, w, c)

        object_aug_update = {
            add_predicted('rope'):          rope_aug,
            add_predicted('left_gripper'):  left_gripper_aug,
            add_predicted('right_gripper'): right_gripper_aug,
            'left_gripper_position':        left_gripper_position_aug,
            'right_gripper_position':       right_gripper_position_aug,
        }

        if DEBUG_VIZ_STATE_AUG:
            stepper = RvizSimpleStepper()
            for b in debug_viz_batch_indices(batch_size):
                env_b = {
                    'env':          inputs['env'][b],
                    'res':          res[b],
                    'extent':       inputs['extent'][b],
                    'origin_point': inputs['origin_point'][b],
                }

                self.plot_environment_rviz(env_b)
                self.debug_viz_state_action(object_aug_update, b, 'aug', color='white')
                stepper.step()
        return object_aug_update, local_origin_point_aug, local_center_aug

    def compute_collision_free_point_ik(self,
                                        default_robot_state,
                                        points,
                                        group_name,
                                        tip_names,
                                        scene_msg,
                                        ik_params):
        pass

    def aug_ik(self,
               inputs_aug: Dict,
               default_robot_positions,
               ik_params: IkParams,
               batch_size: int):
        """

        Args:
            inputs_aug: a dict containing the desired gripper positions as well as the scene_msg and other state info
            default_robot_positions: default robot joint state to seed IK
            batch_size:

        Returns:

        """
        pass

    def plot_aug_points_rviz(self, obj_i: int, obj_points_b_i, label: str, color_map):
        obj_points_b_i_time = tf.reshape(obj_points_b_i, [-1, 8, 3])
        blocks_aug_msg = MarkerArray()
        for t, obj_points_b_i_t in enumerate(obj_points_b_i_time):
            color_t = ColorRGBA(*color_map(t / obj_points_b_i_time.shape[0]))
            color_t.a = 0.2

            block_center = tf.reduce_mean(obj_points_b_i_t, 0)
            block_x_axis = obj_points_b_i_t[3] - obj_points_b_i_t[0]  # [3]
            block_y_axis = obj_points_b_i_t[1] - obj_points_b_i_t[0]  # [3]
            block_z_axis = obj_points_b_i_t[4] - obj_points_b_i_t[0]  # [3]
            block_x_axis_norm, _ = tf.linalg.normalize(block_x_axis)
            block_y_axis_norm, _ = tf.linalg.normalize(block_y_axis)
            block_z_axis_norm, _ = tf.linalg.normalize(block_z_axis)
            block_rot_mat = tf.stack([block_x_axis_norm, block_y_axis_norm, block_z_axis_norm], axis=-1)
            block_quat = quaternion.from_rotation_matrix(block_rot_mat)

            block_aug_msg = Marker()
            block_aug_msg.ns = 'blocks_' + label
            block_aug_msg.id = t + 1000 * obj_i
            block_aug_msg.header.frame_id = 'world'
            block_aug_msg.action = Marker.ADD
            block_aug_msg.type = Marker.CUBE
            block_aug_msg.scale.x = tf.linalg.norm(block_x_axis)
            block_aug_msg.scale.y = tf.linalg.norm(block_y_axis)
            block_aug_msg.scale.z = tf.linalg.norm(block_z_axis)
            block_aug_msg.pose.position.x = block_center[0]
            block_aug_msg.pose.position.y = block_center[1]
            block_aug_msg.pose.position.z = block_center[2]
            block_aug_msg.pose.orientation.x = block_quat[0]
            block_aug_msg.pose.orientation.y = block_quat[1]
            block_aug_msg.pose.orientation.z = block_quat[2]
            block_aug_msg.pose.orientation.w = block_quat[3]
            block_aug_msg.color = color_t

            blocks_aug_msg.markers.append(block_aug_msg)
        self.viz_aug_pub.publish(blocks_aug_msg)

    @staticmethod
    def is_points_key(k):
        return any([
            re.match('obj.*position', k),
            k == 'jaco_arm/primitive_hand/tcp_pos',
        ])

    def make_dm_task(self, params):
        return PlanarPushingCylindersTask(params)

    def __repr__(self):
        return "cylinders"
