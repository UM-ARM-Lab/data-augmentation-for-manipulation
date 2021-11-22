import re
from copy import deepcopy
from typing import Dict

import hjson
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import torch
from pyjacobian_follower import IkParams

from arc_utilities.algorithms import nested_dict_update
from augmentation.aug_opt_utils import get_local_frame
from dm_envs import primitive_hand
from dm_envs.cylinders_task import PlanarPushingCylindersTask
from dm_envs.planar_pushing_scenario import PlanarPushingScenario, ACTION_Z
from dm_envs.planar_pushing_task import ARM_HAND_NAME, ARM_NAME
from link_bot_data.base_collect_dynamics_data import collect_trajectory
from link_bot_data.color_from_kwargs import color_from_kwargs
from link_bot_data.dataset_utils import coerce_types
from link_bot_data.rviz_arrow import rviz_arrow
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from link_bot_pycommon.marker_index_generator import marker_index_generator
from link_bot_pycommon.pycommon import int_frac_to_range
from moonshine.filepath_tools import load_params
from moonshine.geometry import transform_points_3d, xyzrpy_to_matrices, transformation_jacobian, euler_angle_diff
from moonshine.moonshine_utils import repeat_tensor, add_batch, remove_batch
from moonshine.numpify import numpify
from std_msgs.msg import ColorRGBA
from tf import transformations
from visualization_msgs.msg import MarkerArray, Marker

TINV_TEST_OBJ_IDX = 0


def pos_to_vel(pos):
    vel = pos[1:] - pos[:-1]
    vel = np.pad(vel, [[0, 1], [0, 0], [0, 0]], mode='edge')
    return vel


def squeeze_and_get_xy(p):
    return torch.squeeze(p, 2)[:, :, :2]


def cylinders_to_points(positions, res, radius, height):
    """

    Args:
        positions:  [b, m, T, 3]
        res:  [b, T]
        radius:  [b, T]
        height:  [b, T]

    Returns: [b, m, T, n_POINTS, 3]

    """
    m = positions.shape[1]  # m is the number of objects
    sized_points = size_to_points(radius, height, res)  # [b, T, n_points, 3]
    num_points = sized_points.shape[-2]
    sized_points = repeat_tensor(sized_points, m, axis=1, new_axis=True)  # [b, m, T, n_points, 3]
    ones = tf.ones(positions.shape[:-1] + [1])
    positions_homo = tf.expand_dims(tf.concat([positions, ones], axis=-1), -1)  # [b, m, T, 4, 1]
    rot_homo = tf.concat([tf.eye(3), tf.zeros([1, 3])], axis=0)
    rot_homo = repeat_tensor(rot_homo, positions.shape[0], 0, True)
    rot_homo = repeat_tensor(rot_homo, positions.shape[1], 1, True)
    rot_homo = repeat_tensor(rot_homo, positions.shape[2], 2, True)
    transform_matrix = tf.concat([rot_homo, positions_homo], axis=-1)  # [b, m, T, 4, 4]
    transform_matrix = repeat_tensor(transform_matrix, num_points, 3, True)
    obj_points = transform_points_3d(transform_matrix, sized_points)  # [b, m, T, num_points, 3]
    return obj_points


def make_odd(x):
    return tf.where(tf.cast(x % 2, tf.bool), x, x + 1)


NUM_POINTS = 128
cylinder_points_rng = np.random.RandomState(0)


def size_to_points(radius, height, res):
    """

    Args:
        radius: [b, T]
        height: [b, T]
        res: [b, T]

    Returns: [b, T, n_points, 3]

    """
    batch_size, time = radius.shape
    res = res[0, 0]
    radius = radius[0, 0]
    height = height[0, 0]

    n_side = make_odd(tf.cast(2 * radius / res, tf.int64))
    n_height = make_odd(tf.cast(height / res, tf.int64))
    p = tf.linspace(-radius, radius, n_side)
    grid_points = tf.stack(tf.meshgrid(p, p), -1)  # [n_side, n_side, 2]
    in_circle = tf.linalg.norm(grid_points, axis=-1) <= radius
    in_circle_indices = tf.where(in_circle)
    points_in_circle = tf.gather_nd(grid_points, in_circle_indices)
    points_in_circle_w_height = repeat_tensor(points_in_circle, n_height, 0, True)
    z = tf.linspace(0., height, n_height) - height / 2
    z = repeat_tensor(z, points_in_circle_w_height.shape[1], 1, True)[..., None]
    points = tf.concat([points_in_circle_w_height, z], axis=-1)
    points = tf.reshape(points, [-1, 3])

    sampled_points_indices = cylinder_points_rng.randint(0, points.shape[0], NUM_POINTS)
    sampled_points = tf.gather(points, sampled_points_indices, axis=0)

    points_batch = repeat_tensor(sampled_points, batch_size, 0, True)
    points_batch_time = repeat_tensor(points_batch, time, 1, True)
    return points_batch_time


def get_k_with_stats(batch, k):
    v = batch[f"{k}"]
    v_mean = batch[f"{k}/mean"]
    v_std = batch[f"{k}/std"]
    return v, v_mean, v_std


def make_cylinder_marker(color_msg, height, idx, ns, position, radius):
    marker = Marker(ns=ns, action=Marker.ADD, type=Marker.CYLINDER, id=idx, color=color_msg)
    marker.header.frame_id = 'world'
    marker.pose.position.x = position[0, 0]
    marker.pose.position.y = position[0, 1]
    marker.pose.position.z = position[0, 2]
    marker.pose.orientation.w = 1
    marker.scale.x = radius * 2
    marker.scale.y = radius * 2
    marker.scale.z = height

    return marker


def make_vel_arrow(position, velocity, height, color_msg, idx, ns, vel_scale=1.0):
    start = position[0] + np.array([0, 0, height / 2 + 0.0005])
    end = start + velocity[0] * np.array([vel_scale, vel_scale, 1])
    vel_color_factor = 0.4
    vel_color = ColorRGBA(color_msg.r * vel_color_factor,
                          color_msg.g * vel_color_factor,
                          color_msg.b * vel_color_factor,
                          color_msg.a)
    vel_marker = rviz_arrow(start, end,
                            label=ns + 'vel',
                            color=vel_color,
                            idx=idx)
    return vel_marker


class CylindersScenario(PlanarPushingScenario):

    def iter_keys(self, num_objs):
        # NOTE: the robot goes first, this is relied on in aug_apply_no_ik
        yield True, -1, ARM_HAND_NAME
        for obj_idx in range(num_objs):
            obj_k = f'obj{obj_idx}'
            yield False, obj_idx, obj_k

    def iter_positions(self, inputs, num_objs):
        for is_robot, obj_idx, k in self.iter_keys(num_objs):
            if is_robot:
                pos_k = k + '/tcp_pos'
            else:
                pos_k = k + '/position'

            if pos_k in inputs:
                pos = inputs[pos_k]
            else:
                pos = None

            yield is_robot, obj_idx, k, pos_k, pos

    def iter_positions_velocities(self, inputs, num_objs):
        for is_robot, obj_idx, k, pos_k, pos in self.iter_positions(inputs, num_objs):
            if not is_robot:
                vel_k = k + "/linear_velocity"
            else:
                vel_k = k + "/tcp_vel"

            if vel_k in inputs:
                vel = inputs[vel_k]
            else:
                vel = None

            yield is_robot, obj_idx, k, pos_k, vel_k, pos, vel

    def plot_state_rviz(self, state: Dict, **kwargs):
        super().plot_state_rviz(state, **kwargs)

        ns = kwargs.get("label", "")
        idx = kwargs.get("idx", 0)
        color_msg = color_from_kwargs(kwargs, 1.0, 0, 0.0)

        num_objs = state['num_objs'][0]
        height = state['height'][0]
        radius = state['radius'][0]
        msg = MarkerArray()

        ig = marker_index_generator(idx)

        for is_robot, obj_idx, k, pos_k, vel_k, pos, vel in self.iter_positions_velocities(state, num_objs):
            if is_robot:
                if pos is not None:
                    robot_pos_viz = deepcopy(pos)
                    robot_pos_viz[0, 2] += primitive_hand.HALF_HEIGHT
                    robot_color_msg = deepcopy(color_msg)
                    robot_color_msg.b = 1 - robot_color_msg.b
                    marker = make_cylinder_marker(robot_color_msg, primitive_hand.HEIGHT, next(ig), ns + '_robot',
                                                  robot_pos_viz, primitive_hand.RADIUS)
                    msg.markers.append(marker)
                if pos is not None and vel is not None:
                    vel_marker = make_vel_arrow(pos, vel, primitive_hand.HEIGHT + 0.005, color_msg, next(ig),
                                                ns + '_robot')
                    msg.markers.append(vel_marker)
            else:
                if pos is not None:
                    obj_marker = make_cylinder_marker(color_msg, height, next(ig), ns, pos, radius)
                    msg.markers.append(obj_marker)
                if vel is not None:
                    obj_vel_marker = make_vel_arrow(pos, vel, height, color_msg, next(ig), ns)
                    msg.markers.append(obj_vel_marker)

        self.state_viz_pub.publish(msg)

    def compute_obj_points(self, inputs: Dict, num_object_interp: int, batch_size: int):
        """

        Args:
            inputs: contains the poses and size of the blocks, over a whole trajectory, which we convert into points
            num_object_interp:
            batch_size:

        Returns: [b, m_objects, T, n_points, 3]

        """
        height = inputs['height'][:, :, 0]  # [b, T]
        radius = inputs['radius'][:, :, 0]  # [b, T]
        num_objs = inputs['num_objs'][0, 0, 0]  # assumed fixed across batch/time
        positions = []  # [b, m, T, 3]
        for is_robot, obj_idx, k, pos_k, pos in self.iter_positions(inputs, num_objs):
            if not is_robot:
                pos = pos[:, :, 0]  # [b, T, 3]
                positions.append(pos)

        positions = tf.stack(positions, axis=1)
        time = positions.shape[2]

        res = repeat_tensor(inputs['res'], time, 1, True)  # [b]

        obj_points = cylinders_to_points(positions, res=res, radius=radius, height=height)
        robot_radius = repeat_tensor(primitive_hand.RADIUS, batch_size, 0, True)
        robot_radius = repeat_tensor(robot_radius, time, 1, True)
        robot_height = repeat_tensor(primitive_hand.HEIGHT, batch_size, 0, True)
        robot_height = repeat_tensor(robot_height, time, 1, True)
        tcp_positions = tf.reshape(inputs[f'{ARM_HAND_NAME}/tcp_pos'], [batch_size, 1, time, 3])
        robot_cylinder_positions = tcp_positions + [0, 0, primitive_hand.HALF_HEIGHT]
        robot_points = cylinders_to_points(robot_cylinder_positions, res=res, radius=robot_radius, height=robot_height)

        obj_points = tf.concat([robot_points, obj_points], axis=1)

        return obj_points

    def set_state_from_dict(self, predetermined_start_state: Dict):
        for obj in self.task.objs:
            if obj.name in predetermined_start_state:
                obj.set_pose(self.env.physics, position=predetermined_start_state[obj.name])
        self.env.step(np.zeros(self.action_spec.shape))

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
        """

        Args:
            batch: dict of data
            batch_size:
            obj_idx:
            time:
            device:

        Returns:
            obj_attr [b, T, n_attr]
            obj_state [b, T, n_state]
            obj_action [b, T, n_action]

        """
        is_robot = torch.zeros([batch_size, 1], device=device)
        radius = batch['radius'][:, 0]  # assume constant across time
        obj_attr = torch.cat([radius, is_robot], dim=-1)

        obj_pos_k = f"obj{obj_idx}/position"
        obj_pos = batch[obj_pos_k]  # [b, T, 2]
        obj_pos = squeeze_and_get_xy(obj_pos)

        obj_vel_k = f"obj{obj_idx}/linear_velocity"
        obj_vel = batch[obj_vel_k]  # [b, T, 2]
        obj_vel = squeeze_and_get_xy(obj_vel)

        obj_state = torch.cat([obj_pos, obj_vel], dim=-1)  # [b, T, 4]

        obj_action = torch.zeros([batch_size, time - 1, 0], device=device)

        return obj_attr, obj_state, obj_action

    def propnet_robot_v(self, batch, batch_size, time, device):
        is_robot = torch.ones([batch_size, 1], device=device)
        radius = torch.ones([batch_size, 1], device=device) * primitive_hand.RADIUS
        robot_attr = torch.cat([radius, is_robot], dim=-1)

        robot_pos_k = f"{ARM_HAND_NAME}/tcp_pos"
        robot_pos = batch[robot_pos_k]
        robot_pos = squeeze_and_get_xy(robot_pos)

        robot_vel_k = f"{ARM_HAND_NAME}/tcp_vel"
        robot_vel = batch[robot_vel_k]
        robot_vel = squeeze_and_get_xy(robot_vel)

        robot_state = torch.cat([robot_pos, robot_vel], dim=-1)  # [b, T, 4]

        robot_action = torch.zeros([batch_size, time - 1, 0], device=device)

        return robot_attr, robot_state, robot_action

    def propnet_add_vel(self, example: Dict):
        num_objs = example['num_objs'][0, 0]  # assumed fixed across time
        robot_pos = example[f'{ARM_HAND_NAME}/tcp_pos']
        robot_vel = pos_to_vel(robot_pos)
        robot_vel_k = f"{ARM_HAND_NAME}/tcp_vel"
        vel_state_keys = [robot_vel_k]
        example[robot_vel_k] = robot_vel
        for obj_idx in range(num_objs):
            obj_pos = example[f'obj{obj_idx}/position']
            obj_vel = pos_to_vel(obj_pos)
            obj_vel_k = f"obj{obj_idx}/linear_velocity"
            example[obj_vel_k] = obj_vel
            vel_state_keys.append(obj_vel_k)
        return example, vel_state_keys

    def propnet_outputs_to_state(self, inputs, pred_vel, pred_pos, b, t):
        pred_state_t = {}
        height_b_t = inputs['height'][b, t]
        pred_state_t['height'] = height_b_t
        pred_state_t['radius'] = inputs['radius'][b, t]
        num_objs = inputs['num_objs'][b, t, 0]
        pred_state_t['num_objs'] = [num_objs]

        pred_robot_pos_b_t_2d = pred_pos[b, t, 0]
        default_robot_z = torch.zeros(1) * 0.01  # we've lost this info so just put something that will visualize ok
        pred_robot_pos_b_t_3d = torch.cat([pred_robot_pos_b_t_2d, default_robot_z])
        pred_robot_vel_b_t_2d = pred_vel[b, t, 0]
        pred_robot_vel_b_t_3d = torch.cat([pred_robot_vel_b_t_2d, torch.zeros(1)])
        pred_state_t[f'{ARM_HAND_NAME}/tcp_pos'] = torch.unsqueeze(pred_robot_pos_b_t_3d, 0).detach()
        pred_state_t[f'{ARM_HAND_NAME}/tcp_vel'] = torch.unsqueeze(pred_robot_vel_b_t_3d, 0).detach()

        for j in range(num_objs):
            pred_pos_b_t_2d = pred_pos[b, t, j + 1]
            pred_pos_b_t_3d = torch.cat([pred_pos_b_t_2d, height_b_t / 2])
            pred_vel_b_t_2d = pred_vel[b, t, j + 1]
            pred_vel_b_t_3d = torch.cat([pred_vel_b_t_2d, torch.zeros(1)])
            pred_state_t[f'obj{j}/position'] = torch.unsqueeze(pred_pos_b_t_3d, 0).detach()
            pred_state_t[f'obj{j}/linear_velocity'] = torch.unsqueeze(pred_vel_b_t_3d, 0).detach()

        return pred_state_t

    def propnet_rel(self, obj_pos, num_objects, relation_dim, is_close_threshold, device=None):
        """

        Args:
            num_objects: number of objects/particles, $|O|$
            relation_dim: dimension of the relation vector
            is_close_threshold: in meters. Defines whether a relation will exist between two objects
            device:

        Returns:
            Rr: [num_objects, num_relations], binary, 1 at [obj_i,rel_j] means object i is the receiver in relation j
            Rs: [num_objects, num_relations], binary, 1 at [obj_i,rel_j] means object i is the sender in relation j
            Ra: [num_relations, attr_dim] containing the relation attributes

        """
        # we assume the robot is _first_ in the list of objects
        # the robot is included as an object here
        batch_size = obj_pos.shape[0]
        n_rel = num_objects * (num_objects - 1)

        Rs = torch.zeros(batch_size, num_objects, n_rel, device=device)
        Rr = torch.zeros(batch_size, num_objects, n_rel, device=device)
        Ra = torch.zeros(batch_size, n_rel, relation_dim, device=device)  # relation attributes information

        rel_idx = 0
        for sender_idx, receiver_idx in np.ndindex(num_objects, num_objects):
            if sender_idx == receiver_idx:
                continue

            distance = (obj_pos[:, sender_idx] - obj_pos[:, receiver_idx]).square().sum(dim=-1)
            is_close = (distance < is_close_threshold ** 2).float()

            Rs[:, sender_idx, rel_idx] = is_close
            Rr[:, receiver_idx, rel_idx] = is_close

            rel_idx += 1

        return Rs, Rr, Ra

    def initial_identity_aug_params(self, batch_size, k_transforms):
        return tf.zeros([batch_size, k_transforms, 3], tf.float32)  # change in x, y, theta (about z)

    def sample_target_aug_params(self, seed, aug_params, n_samples):
        trans_lim = tf.ones([2]) * aug_params['target_trans_lim']
        trans_distribution = tfp.distributions.Uniform(low=-trans_lim, high=trans_lim)

        theta_lim = tf.ones([1]) * aug_params['target_euler_lim']
        theta_distribution = tfp.distributions.Uniform(low=-theta_lim, high=theta_lim)

        trans_target = trans_distribution.sample(sample_shape=n_samples, seed=seed())
        theta_target = theta_distribution.sample(sample_shape=n_samples, seed=seed())

        target_params = tf.concat([trans_target, theta_target], -1)
        return target_params

    def plot_transform(self, obj_i, transform_params, frame_id):
        """

        Args:
            frame_id:
            transform_params: [x,y,theta]

        Returns:

        """
        target_pos_b = [transform_params[0], transform_params[1], 0]
        theta = transform_params[2]
        target_quat_b = transformations.quaternion_from_euler(0, 0, theta)
        self.tf.send_transform(target_pos_b, target_quat_b, f'aug_opt_initial_{obj_i}', frame_id, False)

    def aug_target_pos(self, target):
        return tf.concat([target[0], target[1], 0], axis=0)

    def transformation_params_to_matrices(self, obj_transforms):
        xy = obj_transforms[..., :2]
        theta = obj_transforms[..., 2:3]
        zrp = tf.zeros(obj_transforms.shape[:-1] + [3])
        xyzrpy = tf.concat([xy, zrp, theta], axis=-1)
        return xyzrpy_to_matrices(xyzrpy)

    def aug_apply_no_ik(self,
                        moved_mask,
                        m,
                        to_local_frame,
                        inputs: Dict,
                        batch_size,
                        time,
                        *args,
                        **kwargs,
                        ):
        """

        Args:
            moved_mask: [b, n_objects]
            m: [b, k, 4, 4]
            to_local_frame: [b, 3]
            inputs:
            batch_size:
            time:

        Returns:

        """
        to_local_frame_expanded1 = to_local_frame[:, None]
        to_local_frame_expanded2 = to_local_frame[:, None, None]
        zeros_expanded2 = tf.zeros([batch_size, 1, 1, 3])
        m_expanded = m[:, None]
        no_translation_mask = np.ones(m_expanded.shape)
        no_translation_mask[..., 0:3, 3] = 0
        m_expanded_no_translation = m_expanded * no_translation_mask

        def _transform(m, points, _to_local_frame):
            points_local_frame = points - _to_local_frame
            points_local_frame_aug = transform_points_3d(m, points_local_frame)
            return points_local_frame_aug + _to_local_frame

        # apply transformations to the state
        num_objs = inputs['num_objs'][0, 0, 0]
        object_aug_update = {
        }

        moved_mask_expanded = moved_mask[:, :, None, None, None]
        for j, pos_vel_data in enumerate(self.iter_positions_velocities(inputs, num_objs)):
            is_robot, obj_idx, k, pos_k, vel_k, pos, vel = pos_vel_data
            pos_aug = _transform(m_expanded, pos, to_local_frame_expanded2)
            mmj = moved_mask_expanded[:, j]
            object_aug_update[pos_k] = pos_aug * mmj + (1 - mmj) * pos
            if vel is not None:
                vel_aug = _transform(m_expanded_no_translation, vel, zeros_expanded2)
                object_aug_update[vel_k] = vel_aug * mmj + (1 - mmj) * vel

        # apply transformations to the action
        gripper_position = inputs['gripper_position']
        gripper_position_aug = _transform(m, gripper_position, to_local_frame_expanded1)
        object_aug_update['gripper_position'] = gripper_position_aug

        if kwargs.get('visualize', False):
            for b in debug_viz_batch_indices(batch_size):
                env_b = {
                    'env':          inputs['env'][b],
                    'res':          inputs['res'][b],
                    'extent':       inputs['extent'][b],
                    'origin_point': inputs['origin_point'][b],
                }
                object_aug_update_viz = deepcopy(object_aug_update)
                object_aug_update_viz.update({
                    'num_objs': inputs['num_objs'],
                    'height':   inputs['height'],
                    'radius':   inputs['radius'],
                })
                object_aug_update_viz_b = {k: v[b] for k, v in object_aug_update_viz.items()}

                self.plot_environment_rviz(env_b)
                for t in range(time):
                    object_aug_update_b_t = {k: v[0] for k, v in object_aug_update_viz_b.items()}
                    object_aug_update_b_t = numpify(object_aug_update_b_t)
                    self.plot_state_rviz(object_aug_update_b_t, label='aug_no_ik', color='white', id=t)

        return object_aug_update, None, None

    def aug_ik(self,
               inputs: Dict,
               inputs_aug: Dict,
               ik_params: IkParams,
               batch_size: int):
        """

        Args:
            inputs:
            inputs_aug: a dict containing the desired gripper positions as well as the scene_msg and other state info
            batch_size:

        Returns:
            is_ik_valid: [b]
            keys

        """
        tcp_pos_aug = inputs_aug[f'{ARM_HAND_NAME}/tcp_pos']

        is_ik_valid = []
        joint_positions_aug = []
        for b in range(batch_size):
            tcp_pos_aug_b = tcp_pos_aug[b]
            is_ik_valid_b = True
            joint_positions_aug_b = []
            for t in range(tcp_pos_aug_b.shape[0]):
                tcp_pos_aug_b_t = tcp_pos_aug_b[t]
                success, joint_position_aug_b_t = self.task.solve_position_ik(self.env.physics, tcp_pos_aug_b_t)
                joint_positions_aug_b.append(joint_position_aug_b_t)
                if not success:
                    is_ik_valid_b = False
            joint_positions_aug.append(joint_positions_aug_b)
            is_ik_valid.append(is_ik_valid_b)

        joint_positions_aug = tf.stack(joint_positions_aug)  # [b, T, 6]
        joint_positions_aug = tf.expand_dims(joint_positions_aug, -2)
        joint_positions_aug_sin = tf.sin(joint_positions_aug)
        joint_positions_aug_cos = tf.cos(joint_positions_aug)
        joint_positions_aug_sincos = tf.stack([joint_positions_aug_sin, joint_positions_aug_cos], -1)  # [b, T, 1, 6, 2]
        joint_positions_aug_sincos = tf.cast(joint_positions_aug_sincos, tf.float32)
        is_ik_valid = tf.cast(tf.stack(is_ik_valid), tf.float32)

        joints_pos_k = f'{ARM_NAME}/joints_pos'

        inputs_aug.update({
            joints_pos_k: joint_positions_aug_sincos,
        })
        return is_ik_valid, [joints_pos_k]

    def aug_transformation_jacobian(self, obj_transforms):
        """

        Args:
            obj_transforms: [b, k_transforms, p]

        Returns: [b, k_transforms, p, 4, 4]

        """
        zrp = tf.zeros(obj_transforms.shape[:-1] + [3])
        xy = obj_transforms[..., :2]
        theta = obj_transforms[..., 2:3]
        xyzrpy = tf.concat([xy, zrp, theta], axis=-1)
        jacobian = transformation_jacobian(xyzrpy)
        jacobian_xy = jacobian[..., 0:2, :, :]
        jacobian_theta = jacobian[..., 2:3, :, :]
        jacobian_xyt = tf.concat([jacobian_xy, jacobian_theta], axis=-3)
        return jacobian_xyt

    def aug_distance(self, transforms1, transforms2):
        trans1 = transforms1[..., :2]
        trans2 = transforms2[..., :2]
        theta1 = transforms1[..., 2:3]
        theta2 = transforms2[..., 2:3]
        theta_dist = tf.linalg.norm(euler_angle_diff(theta1, theta2), axis=-1)
        trans_dist = tf.linalg.norm(trans1 - trans2, axis=-1)
        distances = trans_dist + theta_dist
        max_distance = tf.reduce_max(distances)
        return max_distance

    @staticmethod
    def aug_copy_inputs(inputs):
        aug_copy_keys = [
            'num_objs',
            'radius',
            'height',
            'joint_names',
            'time_idx',
            'dt',
        ]
        return {k: inputs[k] for k in aug_copy_keys}

    @staticmethod
    def aug_merge_hparams(dataset_dir, out_example, outdir):
        in_hparams = load_params(dataset_dir)
        state_keys = in_hparams['data_collection_params']['state_keys']
        aug_state_keys = list(set(out_example.keys()).intersection(state_keys))
        update = {
            'used_augmentation':      True,
            'data_collection_params': {
                'state_keys': aug_state_keys,
            }
        }
        out_hparams = deepcopy(in_hparams)
        nested_dict_update(out_hparams, update)
        with (outdir / 'hparams.hjson').open("w") as out_f:
            hjson.dump(out_hparams, out_f)

    def set_obj_poses(self, obj_positions, obj_quaternions):
        for i, obj in enumerate(self.task.objs):
            obj.set_pose(self.env.physics, position=obj_positions[i], quaternion=obj_quaternions[i])

    def tinv_set_state(self, params, state_rng, visualize):
        obj_z = params['height'] / 2
        # move all the cylinders except for one away
        y_min = params['extent'][2]
        y_max = params['extent'][3]
        x = params['extent'][1] - params['radius']
        obj_positions = [[x, int_frac_to_range(i, len(self.task.objs), y_min, y_max), obj_z] for i, obj in
                         enumerate(self.task.objs)]
        obj_quaternions = [[1, 0, 0, 0] for _ in self.task.objs]
        self.set_obj_poses(obj_positions, obj_quaternions)

        test_obj = self.task.objs[TINV_TEST_OBJ_IDX]
        test_obj.set_pose(self.env.physics, position=[0, 0, obj_z])

        for _ in range(10):
            self.env.step(np.zeros(self.action_spec.shape))

        if visualize:
            state = self.get_state()
            self.plot_state_rviz(state, label='tinv_set_state')

    def tinv_generate_data(self, action_rng: np.random.RandomState, params, visualize):
        example, _ = collect_trajectory(params=params,
                                        scenario=self,
                                        traj_idx=0,
                                        predetermined_start_state=None,
                                        predetermined_actions=None,
                                        verbose=(1 if visualize else 0),
                                        action_rng=action_rng)
        return example

    def tinv_sample_transform(self, transform_sampling_rng: np.random.RandomState, scaling: float):
        trans = 1.0
        euler = np.pi
        lim = np.array([trans, trans, trans, euler, euler, euler]) * scaling
        transform = transform_sampling_rng.uniform(-lim, lim)
        return transform

    def tinv_apply_transformation(self, example: Dict, transform, visualize):
        # apply the se3 transformation to the positions of everything

        num_objs = example['num_objs'][0][0]
        time = example['time_idx'].shape[0]

        example = coerce_types(example)
        example_batch = add_batch(example)
        moved_mask = np.zeros([1, num_objs + 1], np.float32)
        moved_mask[:, 0] = 1  # the robot always moves
        moved_mask[:, TINV_TEST_OBJ_IDX + 1] = 1
        example_aug = deepcopy(example)
        # FIXME: we should be using self.transformation_params_to_matrices, but that currently only does x,y,yaw.
        #  once the invariance learning is working, we should make the scenario do full xyzrpy and change this part
        m = xyzrpy_to_matrices(tf.convert_to_tensor(add_batch(transform), tf.float32))
        obj_points = self.compute_obj_points(example_batch, num_object_interp=1, batch_size=1)
        to_local_frame = get_local_frame(moved_mask, obj_points)
        example_aug_update, _, _ = self.aug_apply_no_ik(moved_mask, m, to_local_frame, example_batch,
                                                        batch_size=1, time=time, visualize=visualize)
        example_aug_update = remove_batch(example_aug_update)
        example_aug = nested_dict_update(example_aug, example_aug_update)

        return example_aug

    def tinv_error(self, example: Dict, example_aug: Dict):
        num_objs = example['num_objs'][0][0]
        error = 0
        for is_robot, obj_idx, k, pos_k, pos in self.iter_positions(example, num_objs):
            if is_robot:
                continue
            # compute the per-object error and add it
            pos_aug = example_aug[pos_k]
            obj_error = tf.reduce_mean(tf.square(pos - pos_aug))
            error += obj_error

        return error

    def tinv_viz(self, example_aug: Dict, label='aug', color='b'):
        pass
        # for t in range(None):
        #     state_t = index_t(states, t)
        #     self.plot_state_rviz(state_t, label=f'{label}_{t}')

    def tinv_set_state_from_aug_pred(self, example_aug_pred, visualize):
        for i, obj in enumerate(self.task.objs):
            # [0,0] because t=0 and extra dim of 1
            pos = example_aug_pred[f'obj{i}/position'][0, 0]
            quat = example_aug_pred[f'obj{i}/orientation'][0, 0]
            obj.set_pose(self.env.physics, position=pos, quaternion=quat)

        aug_tcp_pos = example_aug_pred[f'{ARM_HAND_NAME}/tcp_pos'][0, 0]
        success, joint_position_aug = self.task.solve_position_ik(self.env.physics, aug_tcp_pos)

        if not success:
            raise RuntimeError(f"failed to solve ik to {aug_tcp_pos}")

        # teleports arm joints
        for joint_idx, joint_value in enumerate(joint_position_aug):
            self.env.physics.named.data.qpos[f'{ARM_NAME}/joint_{joint_idx + 1}'] = joint_value

        for _ in range(10):
            self.env.step(np.zeros(self.action_spec.shape))

        if visualize:
            state = self.get_state()
            self.plot_state_rviz(state, label='tinv_set_state')

    def tinv_generate_data_from_aug_pred(self, example_aug_pred, visualize):
        predetermined_actions = example_aug_pred['gripper_position']
        example_aug_actual, _ = collect_trajectory(params=params,
                                                   scenario=self,
                                                   traj_idx=0,
                                                   predetermined_start_state=None,
                                                   predetermined_actions=predetermined_actions,
                                                   verbose=(1 if visualize else 0),
                                                   action_rng=action_rng)
        return example_aug_actual
