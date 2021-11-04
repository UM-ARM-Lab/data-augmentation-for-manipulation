import re
from typing import Dict

import tensorflow as tf
import torch
from pyjacobian_follower import IkParams
from tensorflow_graphics.geometry.transformation import quaternion

from dm_envs.cylinders_task import PlanarPushingCylindersTask
from dm_envs.planar_pushing_scenario import PlanarPushingScenario
from dm_envs.planar_pushing_task import ARM_HAND_NAME
from link_bot_data.color_from_kwargs import color_from_kwargs
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker


def squeeze_and_get_xy(p):
    return torch.squeeze(p, 2)[:, :, :2]


def cylinders_to_points(positions, height, radius):
    """

    Args:
        positions:  [b, m, T, 3]
        height:  [b, T]
        radius:  [b, T]

    Returns: [b, m, T, 8, 3]

    """
    raise NotImplementedError()
    # m = positions.shape[1]
    # sized_points = size_to_points(height, radius)  # [b, T, 8, 3]
    # sized_points = repeat_tensor(sized_points, m, axis=1, new_axis=True)  # [b, m, T, 8, 3]
    # transform_matrix = transformation_matrices_from_pos_quat(positions, [0, 0, 0, 1])  # [b, m, T, 4, 4]
    # transform_matrix = repeat_tensor(transform_matrix, 8, 3, True)
    # obj_points = transform_points_3d(transform_matrix, sized_points)  # [b, m, T, 8, 3]
    # return obj_points


def size_to_points(height, radius):
    """

    Args:
        height: [b, T]
        radius: [b, T]

    Returns: [b, T, 8, 3] where 8 represents the corners the cube, and 3 is x,y,z.

    """
    raise NotImplementedError()
    # unit_cylinder_points = tf.constant([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
    #                                     [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]], tf.float32) - 0.5
    # return unit_cylinder_points[None, None] * size[:, :, None, None]  # [b, T, 8, 3]


def get_k_with_stats(batch, k):
    v = batch[f"{k}"]
    v_mean = batch[f"{k}/mean"]
    v_std = batch[f"{k}/std"]
    return v, v_mean, v_std


class CylindersScenario(PlanarPushingScenario):

    def plot_state_rviz(self, state: Dict, **kwargs):
        super().plot_state_rviz(state, **kwargs)

        ns = kwargs.get("label", "")
        idx = kwargs.get("idx", 0)
        color_msg = color_from_kwargs(kwargs, 1.0, 0, 0.0)

        num_objs = state['num_objs'][0]
        height = state['height'][0]
        radius = state['radius'][0]
        msg = MarkerArray()
        for i in range(num_objs):
            position = state[f'obj{i}/position']

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
            marker.pose.orientation.w = 1
            marker.scale.x = radius * 2
            marker.scale.y = radius * 2
            marker.scale.z = height
            msg.markers.append(marker)

            velocity = state[f'obj{i}/linear_velocity'] * 20
            end = position[0] + velocity[0]
            vel_marker = rviz_arrow(position[0], end, label=ns + 'vel', color=color_msg, idx=idx * num_objs + i)
            msg.markers.append(vel_marker)

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
        for i in range(num_objs):
            pos = inputs[f"obj{i}/position"][:, :, 0]  # [b, T, 3]
            # in our mujoco dataset the quaternions are stored w,x,y,z but the rest of our code assumes xyzw
            positions.append(pos)
        positions = tf.stack(positions, axis=1)

        obj_points = cylinders_to_points(positions, height, radius)

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
        raise NotImplementedError()
        # """
        #
        # Args:
        #     m: [b, 4, 4]
        #     to_local_frame: [b, 1, 3]  the 1 can also be equal to time
        #     inputs:
        #     batch_size:
        #     time:
        #     h:
        #     w:
        #     c:
        #
        # Returns:
        #
        # """
        # # apply those to the rope and grippers
        # rope_points = tf.reshape(inputs[add_predicted('rope')], [batch_size, time, -1, 3])
        # left_gripper_point = inputs[add_predicted('left_gripper')]
        # right_gripper_point = inputs[add_predicted('right_gripper')]
        # left_gripper_points = tf.expand_dims(left_gripper_point, axis=-2)
        # right_gripper_points = tf.expand_dims(right_gripper_point, axis=-2)
        #
        # def _transform(m, points, _to_local_frame):
        #     points_local_frame = points - _to_local_frame
        #     points_local_frame_aug = transform_points_3d(m, points_local_frame)
        #     return points_local_frame_aug + _to_local_frame
        #
        # # m is expanded to broadcast across batch & num_points dimensions
        # rope_points_aug = _transform(m[:, None, None], rope_points, to_local_frame[:, None])
        # left_gripper_points_aug = _transform(m[:, None, None], left_gripper_points, to_local_frame[:, None])
        # right_gripper_points_aug = _transform(m[:, None, None], right_gripper_points, to_local_frame[:, None])
        #
        # # compute the new action
        # left_gripper_position = inputs['left_gripper_position']
        # right_gripper_position = inputs['right_gripper_position']
        # # m is expanded to broadcast across batch dimensions
        # left_gripper_position_aug = _transform(m[:, None], left_gripper_position, to_local_frame)
        # right_gripper_position_aug = _transform(m[:, None], right_gripper_position, to_local_frame)
        #
        # rope_aug = tf.reshape(rope_points_aug, [batch_size, time, -1])
        # left_gripper_aug = tf.reshape(left_gripper_points_aug, [batch_size, time, -1])
        # right_gripper_aug = tf.reshape(right_gripper_points_aug, [batch_size, time, -1])
        #
        # # Now that we've updated the state/action in inputs, compute the local origin point
        # state_aug_0 = {
        #     'left_gripper':  left_gripper_aug[:, 0],
        #     'right_gripper': right_gripper_aug[:, 0],
        #     'rope':          rope_aug[:, 0]
        # }
        # local_center_aug = self.local_environment_center_differentiable(state_aug_0)
        # res = inputs['res']
        # local_origin_point_aug = batch_center_res_shape_to_origin_point(local_center_aug, res, h, w, c)
        #
        # object_aug_update = {
        #     add_predicted('rope'):          rope_aug,
        #     add_predicted('left_gripper'):  left_gripper_aug,
        #     add_predicted('right_gripper'): right_gripper_aug,
        #     'left_gripper_position':        left_gripper_position_aug,
        #     'right_gripper_position':       right_gripper_position_aug,
        # }
        #
        # if DEBUG_VIZ_STATE_AUG:
        #     stepper = RvizSimpleStepper()
        #     for b in debug_viz_batch_indices(batch_size):
        #         env_b = {
        #             'env':          inputs['env'][b],
        #             'res':          res[b],
        #             'extent':       inputs['extent'][b],
        #             'origin_point': inputs['origin_point'][b],
        #         }
        #
        #         self.plot_environment_rviz(env_b)
        #         self.debug_viz_state_action(object_aug_update, b, 'aug', color='white')
        #         stepper.step()
        # return object_aug_update, local_origin_point_aug, local_center_aug

    def compute_collision_free_point_ik(self,
                                        default_robot_state,
                                        points,
                                        group_name,
                                        tip_names,
                                        scene_msg,
                                        ik_params):
        raise NotImplementedError()

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
        raise NotImplementedError()

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
            k == f'{ARM_HAND_NAME}/tcp_pos',
        ])

    def make_dm_task(self, params):
        return PlanarPushingCylindersTask(params)

    def __repr__(self):
        return "cylinders"

    def propnet_obj_v(self, batch, batch_size, obj_idx, time, device):
        obj_attr = torch.zeros([batch_size, 1]).to(device)
        obj_attr_mean = torch.zeros_like(obj_attr)
        obj_attr_std = torch.ones_like(obj_attr)

        obj_pos_k = f"obj{obj_idx}/position"
        obj_pos, obj_pos_mean, obj_pos_std = get_k_with_stats(batch, obj_pos_k)
        obj_pos = squeeze_and_get_xy(obj_pos)
        obj_pos_mean = squeeze_and_get_xy(obj_pos_mean)
        obj_pos_std = squeeze_and_get_xy(obj_pos_std)

        obj_vel_k = f"obj{obj_idx}/linear_velocity"
        obj_vel, obj_vel_mean, obj_vel_std = get_k_with_stats(batch, obj_vel_k)
        obj_vel = squeeze_and_get_xy(obj_vel)
        obj_vel_mean = squeeze_and_get_xy(obj_vel_mean)
        obj_vel_std = squeeze_and_get_xy(obj_vel_std)

        obj_state = torch.cat([obj_pos, obj_vel])
        obj_state_mean = torch.cat([obj_pos_mean, obj_vel_mean])
        obj_state_std = torch.cat([obj_pos_std, obj_vel_std])

        obj_action = torch.zeros([batch_size, time - 1, 3]).to(device)
        obj_action_mean = torch.zeros_like(obj_action)
        obj_action_std = torch.ones_like(obj_action)

        return torch.stack([obj_attr, obj_attr_mean, obj_attr_std]), \
               torch.stack([obj_state, obj_state_mean, obj_state_std]), \
               torch.stack([obj_action, obj_action_mean, obj_action_std])

    def propnet_robot_v(self, batch, batch_size, device):
        robot_attr = torch.ones([batch_size, 1]).to(device)
        robot_attr_mean = torch.zeros_like(robot_attr)
        robot_attr_std = torch.ones_like(robot_attr)

        robot_pos_k = f"{ARM_HAND_NAME}/tcp_pos"
        robot_pos, robot_pos_mean, robot_pos_std = get_k_with_stats(batch, robot_pos_k)
        robot_pos = squeeze_and_get_xy(robot_pos)
        robot_pos_mean = squeeze_and_get_xy(robot_pos_mean)
        robot_pos_std = squeeze_and_get_xy(robot_pos_std)

        robot_vel_k = f"{ARM_HAND_NAME}/tcp_vel"
        robot_vel, robot_vel_mean, robot_vel_std = get_k_with_stats(batch, robot_vel_k)
        robot_vel = squeeze_and_get_xy(robot_vel)
        robot_vel_mean = squeeze_and_get_xy(robot_vel_mean)
        robot_vel_std = squeeze_and_get_xy(robot_vel_std)

        robot_state = torch.cat([robot_pos, robot_vel])
        robot_state_mean = torch.cat([robot_pos_mean, robot_vel_mean])
        robot_state_std = torch.cat([robot_pos_std, robot_vel_std])

        robot_action_k = 'gripper_position'
        robot_action, robot_action_mean, robot_action_std = get_k_with_stats(batch, robot_action_k)

        return torch.stack([robot_attr, robot_attr_mean, robot_attr_std]), \
               torch.stack([robot_state, robot_state_mean, robot_state_std]), \
               torch.stack([robot_action, robot_action_mean, robot_action_std])
