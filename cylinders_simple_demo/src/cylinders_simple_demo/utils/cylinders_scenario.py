from copy import deepcopy
from typing import Dict

import hjson
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

from cylinders_simple_demo.utils.torch_geometry import transform_points_3d, transformation_jacobian, euler_angle_diff, \
    xyzrpy_to_matrices
from cylinders_simple_demo.utils.utils import nested_dict_update, load_params

HEIGHT = 0.08
HALF_HEIGHT = HEIGHT / 2
RADIUS = 0.02
ARM_NAME = 'jaco_arm'
HAND_NAME = 'primitive_hand'
ARM_HAND_NAME = f'{ARM_NAME}/{HAND_NAME}'
ARM_OFFSET = (0., 0.3, 0.)


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
    sized_points = torch.stack(m * [sized_points], 1)  # [b, m, T, n_points, 3]
    ones = torch.ones(positions.shape[:-1] + (1,))
    positions_homo = torch.cat([positions, ones], -1)[..., None]  # [b, m, T, 4, 1]
    rot_homo = torch.cat([torch.eye(3), torch.zeros([1, 3])], 0)
    rot_homo = rot_homo.repeat(positions.shape[:-1] + (1, 1))
    transform_matrix = torch.cat([rot_homo, positions_homo], -1)  # [b, m, T, 4, 4]
    transform_matrix = torch.stack(num_points * [transform_matrix], 3)
    obj_points = transform_points_3d(transform_matrix, sized_points)  # [b, m, T, num_points, 3]
    return obj_points


def make_odd(x):
    return torch.where((x % 2).bool(), x, x + 1)


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

    n_side = make_odd((2 * radius / res).long())
    n_height = make_odd((height / res).long())
    p = torch.linspace(-radius, radius, n_side)
    grid_points = torch.stack(torch.meshgrid(p, p, indexing='xy'), -1)  # [n_side, n_side, 2]
    in_circle = grid_points.norm(dim=-1) <= radius
    in_circle_indices = torch.where(in_circle)
    points_in_circle = grid_points[in_circle_indices]
    points_in_circle_w_height = torch.stack(n_height * [points_in_circle], 0)
    z = torch.linspace(0., height, n_height) - height / 2
    z = torch.stack(points_in_circle_w_height.shape[1] * [z], 1)[..., None]
    points = torch.concat([points_in_circle_w_height, z], dim=-1)
    points = points.reshape([-1, 3])

    sampled_points_indices = torch.tensor(cylinder_points_rng.randint(0, points.shape[0], NUM_POINTS))
    sampled_points = torch.index_select(points, dim=0, index=sampled_points_indices)

    points_batch = torch.stack(batch_size * [sampled_points], 0)
    points_batch_time = torch.stack(time * [points_batch], 1)
    return points_batch_time


def pos_in_bounds(tcp_pos_aug_b):
    s = 0.2
    lower = torch.Tensor([-s, -s, 0])
    upper = torch.Tensor([s, s, 0.01])
    in_bounds = torch.logical_and(lower < tcp_pos_aug_b, tcp_pos_aug_b < upper)
    always_in_bounds = torch.all(in_bounds)
    return always_in_bounds


class CylindersScenario:

    def __init__(self):
        pass

    def iter_keys(self, num_objs):
        # NOTE: the robot goes first, this is relied on in aug_apply_no_ik
        yield True, -1, ARM_HAND_NAME
        for obj_idx in range(num_objs):
            obj_k = f'obj{obj_idx}'
            yield False, obj_idx, obj_k

    def iter_keys_pos_vel(self, num_objs):
        yield ARM_HAND_NAME + '/tcp_pos', ARM_HAND_NAME + '/tcp_vel'
        for obj_idx in range(num_objs):
            obj_k = f'obj{obj_idx}'
            yield obj_k + '/position', obj_k + "/linear_velocity"

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

    def compute_obj_points(self, inputs: Dict, batch_size: int):
        """

        Args:
            inputs: contains the poses and size of the blocks, over a whole trajectory, which we convert into points
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

        positions = torch.stack(positions, dim=1)
        time = positions.shape[2]

        res = torch.stack(time * [inputs['res']], 1)

        obj_points = cylinders_to_points(positions, res=res, radius=radius, height=height)
        robot_radius = torch.tensor(batch_size * [RADIUS])
        robot_radius = torch.stack(time * [robot_radius], 1)
        robot_height = torch.tensor(batch_size * [HEIGHT])
        robot_height = torch.stack(time * [robot_height], 1)
        tcp_positions = inputs[f'{ARM_HAND_NAME}/tcp_pos'].reshape([batch_size, 1, time, 3])
        robot_cylinder_positions = tcp_positions + torch.tensor([0, 0, HALF_HEIGHT])
        robot_points = cylinders_to_points(robot_cylinder_positions, res=res, radius=robot_radius, height=robot_height)

        obj_points = torch.cat([robot_points, obj_points], 1)

        return obj_points

    def __repr__(self):
        return "cylinders"

    def example_to_gif(self, example):
        time = example['time_idx'].shape[0]
        num_objs = example['num_objs'][0, 0]

        # FIXME: de-duplicate this with the dataset visualization code

        fig = plt.figure()
        ax = plt.gca()
        ax.set_aspect("equal")
        s = 0.15
        ax.set_xlim([-s, s])
        ax.set_ylim([-s, s])

        def _func(t):
            # https://stackoverflow.com/questions/49791848/matplotlib-remove-all-patches-from-figure
            for p in reversed(ax.patches):
                p.remove()

            for is_robot, obj_idx, k, pos_k, pos in self.iter_positions(example, num_objs):
                radius = example['radius'][t, 0]
                x = pos[t, 0, 0]
                y = pos[t, 0, 1]
                if is_robot:
                    p = Circle((x, y), RADIUS, color=[1, 0, 1])
                else:
                    p = Circle((x, y), radius, color='red')
                ax.add_patch(p)

        anim = FuncAnimation(fig=fig, func=_func, frames=time)
        return anim

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

        return obj_attr, obj_state

    def propnet_robot_v(self, batch, batch_size, time, device):
        is_robot = torch.ones([batch_size, 1], device=device)
        radius = torch.ones([batch_size, 1], device=device) * RADIUS
        robot_attr = torch.cat([radius, is_robot], dim=-1)

        robot_pos_k = f"{ARM_HAND_NAME}/tcp_pos"
        robot_pos = batch[robot_pos_k]
        robot_pos = squeeze_and_get_xy(robot_pos)

        robot_vel_k = f"{ARM_HAND_NAME}/tcp_vel"
        robot_vel = batch[robot_vel_k]
        robot_vel = squeeze_and_get_xy(robot_vel)

        robot_state = torch.cat([robot_pos, robot_vel], dim=-1)  # [b, T, 4]

        return robot_attr, robot_state

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

    def propnet_outputs_to_state(self, inputs, pred_vel, pred_pos, b, t, obj_dz=0):
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
            pred_pos_b_t_3d = torch.cat([pred_pos_b_t_2d, height_b_t / 2 + obj_dz])
            pred_vel_b_t_2d = pred_vel[b, t, j + 1]
            pred_vel_b_t_3d = torch.cat([pred_vel_b_t_2d, torch.zeros(1)])
            pred_state_t[f'obj{j}/position'] = torch.unsqueeze(pred_pos_b_t_3d, 0).detach()
            pred_state_t[f'obj{j}/linear_velocity'] = torch.unsqueeze(pred_vel_b_t_3d, 0).detach()

        return pred_state_t

    def propnet_rel(self, obj_pos, num_objects, relation_dim, is_close_threshold, device=None):
        """

        Args:
            obj_pos:
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
        return torch.zeros([batch_size, k_transforms, 3])  # change in x, y, theta (about z)

    def sample_target_aug_params(self, rng: np.random.RandomState, aug_params, n_samples):
        trans_lim = torch.ones([2]) * aug_params['target_trans_lim']
        trans_target = torch.Tensor(rng.uniform(low=-trans_lim, high=trans_lim, size=[n_samples, 2]))

        theta_lim = torch.ones([1]) * aug_params['target_euler_lim']
        theta_target = torch.Tensor(rng.uniform(low=-theta_lim, high=theta_lim, size=[n_samples, 1]))

        target_params = torch.cat([trans_target, theta_target], -1)
        return target_params

    def aug_target_pos(self, target):
        return torch.cat([target[0], target[1], 0], 0)

    def transformation_params_to_matrices(self, obj_transforms):
        xy = obj_transforms[..., :2]
        theta = obj_transforms[..., 2:3]
        zrp = torch.zeros(obj_transforms.shape[:-1] + (3,))
        xyzrpy = torch.cat([xy, zrp, theta], -1)
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
        zeros_expanded2 = torch.zeros([batch_size, 1, 1, 3])
        m_expanded = m[:, None]
        no_translation_mask = torch.ones(m_expanded.shape)
        no_translation_mask[..., 0:3, 3] = 0
        m_expanded_no_translation = m_expanded * no_translation_mask

        def _transform(_m, points, _to_local_frame):
            points_local_frame = points - _to_local_frame
            points_local_frame_aug = transform_points_3d(_m, points_local_frame)
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

        return object_aug_update, None, None

    def aug_ik(self,
               inputs: Dict,
               inputs_aug: Dict,
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
        for b in range(batch_size):
            tcp_pos_aug_b = tcp_pos_aug[b]
            is_ik_valid_b = pos_in_bounds(tcp_pos_aug_b)
            is_ik_valid.append(is_ik_valid_b)

        is_ik_valid = torch.stack(is_ik_valid).float()

        return is_ik_valid, []

    def aug_transformation_jacobian(self, obj_transforms):
        """

        Args:
            obj_transforms: [b, k_transforms, p]

        Returns: [b, k_transforms, p, 4, 4]

        """
        zrp = torch.zeros(obj_transforms.shape[:-1] + (3,))
        xy = obj_transforms[..., :2]
        theta = obj_transforms[..., 2:3]
        xyzrpy = torch.cat([xy, zrp, theta], -1)
        jacobian = transformation_jacobian(xyzrpy)
        jacobian_xy = jacobian[..., 0:2, :, :]
        jacobian_theta = jacobian[..., 2:3, :, :]
        jacobian_xyt = torch.cat([jacobian_xy, jacobian_theta], -3)
        return jacobian_xyt

    def aug_distance(self, transforms1, transforms2):
        trans1 = transforms1[..., :2]
        trans2 = transforms2[..., :2]
        theta1 = transforms1[..., 2:3]
        theta2 = transforms2[..., 2:3]
        theta_dist = euler_angle_diff(theta1, theta2).norm(dim=-1)
        trans_dist = (trans1 - trans2).norm(dim=-1)
        distances = trans_dist + theta_dist
        max_distance = distances.max()
        return max_distance

    @staticmethod
    def aug_copy_inputs(inputs):
        aug_copy_keys = [
            'num_objs',
            'radius',
            'height',
            'joint_names',
            'time_idx',
            'traj_idx',
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

    def example_dict_to_flat_vector(self, example):
        num_objs = example['num_objs'][0, 0, 0]
        posvels = []
        for is_robot, obj_idx, k, pos_k, vel_k, pos, vel in self.iter_positions_velocities(example, num_objs):
            posvel = torch.cat([pos, vel], dim=-1)
            posvels.append(posvel)
        posvels = torch.stack(posvels, dim=1)  # [batch_size, m_objects, horizon, 1, pos_vel_dim]
        batch_size = posvels.shape[0]
        return torch.reshape(posvels, [batch_size, -1])

    def flat_vector_to_example_dict(self, example, flat_vector_aug):
        pos_vel_dim = 6
        num_objs = example['num_objs'][0, 0, 0]
        time = example['num_objs'].shape[1]
        batch_size = example['num_objs'].shape[0]
        matrix_aug = flat_vector_aug.reshape([batch_size, num_objs + 1, time, 1, pos_vel_dim])
        example_aug = deepcopy(example)
        for i, (pos_k, vel_k) in enumerate(self.iter_keys_pos_vel(num_objs)):
            v = matrix_aug[:, i]
            pos = v[..., :3]
            vel = v[..., 3:]
            example_aug[pos_k] = pos.cpu().detach().numpy()
            example_aug[vel_k] = vel.cpu().detach().numpy()
        return example_aug
