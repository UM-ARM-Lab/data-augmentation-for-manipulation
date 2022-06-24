import colorsys
from copy import deepcopy
from typing import Dict

import hjson
import matplotlib.colors as mc
import numpy as np
import torch
from matplotlib import pyplot as plt, patches
from matplotlib.animation import FuncAnimation

from cylinders_simple_demo.utils.torch_geometry import transform_points_3d, transformation_jacobian, euler_angle_diff, \
    xyzrpy_to_matrices, homogeneous
from cylinders_simple_demo.utils.utils import nested_dict_update, load_params

HEIGHT = 0.08
HALF_HEIGHT = HEIGHT / 2
RADIUS = 0.02
ARM_NAME = 'jaco_arm'
HAND_NAME = 'primitive_hand'
ARM_HAND_NAME = f'{ARM_NAME}/{HAND_NAME}'
ARM_OFFSET = (0., 0.3, 0.)
NUM_POINTS = 128
cylinder_points_rng = np.random.RandomState(0)


def adjust_lightness(color, amount=1.0):
    """
    Adjusts the brightness/lightness of a color
    Args:
        color: any valid matplotlib color, could be hex string, or tuple, etc.
        amount: 1 means no change, less than 1 is darker, more than 1 is brighter

    Returns:

    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def pos_to_vel(pos):
    vel = pos[1:] - pos[:-1]
    vel = np.pad(vel, [[0, 1], [0, 0], [0, 0]], mode='edge')
    return vel


def squeeze_and_get_xy(p):
    return torch.squeeze(p, 2)[:, :, :2]


def cylinders_to_points(positions, res, radius, height, device):
    """

    Args:
        positions:  [b, m, T, 3]
        res:  [b, T]
        radius:  [b, T]
        height:  [b, T]

    Returns: [b, m, T, n_POINTS, 3]

    """
    m = positions.shape[1]  # m is the number of objects
    sized_points = size_to_points(radius, height, res, device)  # [b, T, n_points, 3]
    num_points = sized_points.shape[-2]
    sized_points = torch.stack(m * [sized_points], 1)  # [b, m, T, n_points, 3]
    ones = torch.ones(positions.shape[:-1] + (1,), device=device)
    positions_homo = torch.cat([positions, ones], -1)[..., None]  # [b, m, T, 4, 1]
    rot_homo = torch.cat([torch.eye(3, device=device), torch.zeros([1, 3], device=device)], 0)
    rot_homo = rot_homo.repeat(positions.shape[:-1] + (1, 1))
    transform_matrix = torch.cat([rot_homo, positions_homo], -1)  # [b, m, T, 4, 4]
    transform_matrix = torch.stack(num_points * [transform_matrix], 3)
    obj_points = transform_points_3d(transform_matrix, sized_points)  # [b, m, T, num_points, 3]
    return obj_points


def make_odd(x):
    return torch.where((x % 2).bool(), x, x + 1)


def size_to_points(radius, height, res, device):
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
    p = torch.linspace(-radius, radius, n_side, device=device)
    grid_points = torch.stack(torch.meshgrid(p, p, indexing='xy'), -1)  # [n_side, n_side, 2]
    in_circle = grid_points.norm(dim=-1) <= radius
    in_circle_indices = torch.where(in_circle)
    points_in_circle = grid_points[in_circle_indices]
    points_in_circle_w_height = torch.stack(n_height * [points_in_circle], 0)
    z = torch.linspace(0., height, n_height, device=device) - height / 2
    z = torch.stack(points_in_circle_w_height.shape[1] * [z], 1)[..., None]
    points = torch.concat([points_in_circle_w_height, z], dim=-1)
    points = points.reshape([-1, 3])

    sampled_points_indices = torch.from_numpy(cylinder_points_rng.randint(0, points.shape[0], NUM_POINTS)).to(device)
    sampled_points = torch.index_select(points, dim=0, index=sampled_points_indices)

    points_batch = torch.stack(batch_size * [sampled_points], 0)
    points_batch_time = torch.stack(time * [points_batch], 1)
    return points_batch_time


def pos_in_bounds(tcp_pos_aug_b):
    s = 0.2
    lower = torch.tensor([-s, -s, 0], device=tcp_pos_aug_b.device)
    upper = torch.tensor([s, s, 0.01], device=tcp_pos_aug_b.device)
    in_bounds = torch.logical_and(lower < tcp_pos_aug_b, tcp_pos_aug_b < upper)
    always_in_bounds = torch.all(in_bounds)
    return always_in_bounds


class CylindersScenario:

    def __init__(self):
        pass

    @staticmethod
    def iter_keys(num_objs):
        # NOTE: the robot goes first, this is relied on in aug_apply_no_ik
        yield True, -1, ARM_HAND_NAME
        for obj_idx in range(num_objs):
            obj_k = f'obj{obj_idx}'
            yield False, obj_idx, obj_k

    @staticmethod
    def iter_keys_pos_vel(num_objs):
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

    def compute_obj_points(self, inputs: Dict, batch_size: int, device):
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

        obj_points = cylinders_to_points(positions, res=res, radius=radius, height=height, device=device)
        robot_radius = torch.tensor(batch_size * [RADIUS]).to(device)
        robot_radius = torch.stack(time * [robot_radius], 1)
        robot_height = torch.tensor(batch_size * [HEIGHT], device=device)
        robot_height = torch.stack(time * [robot_height], 1)
        tcp_positions = inputs[f'{ARM_HAND_NAME}/tcp_pos'].reshape([batch_size, 1, time, 3])
        robot_cylinder_positions = tcp_positions + torch.tensor([0, 0, HALF_HEIGHT], device=device)
        robot_points = cylinders_to_points(robot_cylinder_positions, res=res, radius=robot_radius, height=robot_height,
                                           device=device)

        obj_points = torch.cat([robot_points, obj_points], 1)

        return obj_points

    def __repr__(self):
        return "cylinders"

    def example_and_predictions_to_animation(self, example, gt_vel, gt_pos, pred_vel, pred_pos):
        fig = plt.figure()
        plt.axis("equal")
        plt.title(f"Trajectory #{example['traj_idx']}")
        ax = plt.gca()
        ax.set_xlim([-.2, .2])

        def viz_t(t):
            while len(ax.patches) > 0:
                ax.patches.pop()

            pred_dict = self.propnet_outputs_to_state(example, pred_vel, pred_pos)

            self.cylinders_viz_t(ax, example, t, objs_color='red', robot_color='black')
            self.cylinders_viz_t(ax, pred_dict, t, objs_color='blue', robot_color='black')

            ax.set_ylim([-.2, .2])

        anim = FuncAnimation(fig, viz_t, frames=50, interval=2)
        return anim

    def cylinders_viz_t(self, ax, example, t, robot_color, objs_color):
        radius = example['radius'][t, 0]
        x = example['jaco_arm/primitive_hand/tcp_pos'][t, 0, 0]
        y = example['jaco_arm/primitive_hand/tcp_pos'][t, 0, 1]
        dx = example['jaco_arm/primitive_hand/tcp_vel'][t, 0, 0]
        dy = example['jaco_arm/primitive_hand/tcp_vel'][t, 0, 1]
        robot = patches.Circle((x, y), radius, color=adjust_lightness(robot_color))
        robot_vel = patches.Arrow(x, y, dx, dy, width=0.01, color=robot_color)
        ax.add_patch(robot)
        ax.add_patch(robot_vel)
        for object_idx in range(9):
            x = example[f'obj{object_idx}/position'][t, 0, 0]
            y = example[f'obj{object_idx}/position'][t, 0, 1]
            dx = example[f'obj{object_idx}/linear_velocity'][t, 0, 0]
            dy = example[f'obj{object_idx}/linear_velocity'][t, 0, 1]

            obj = plt.Circle((x, y), radius, color=adjust_lightness(objs_color))
            obj_vel = plt.Arrow(x, y, dx, dy, width=0.01, color=objs_color)
            ax.add_patch(obj)
            ax.add_patch(obj_vel)

    def example_to_animation(self, example):
        fig = plt.figure()
        plt.axis("equal")
        plt.title(f"Trajectory #{example['traj_idx']}")
        ax = plt.gca()
        ax.set_xlim([-.2, .2])

        def viz_t(t):
            while len(ax.patches) > 0:
                ax.patches.pop()
            self.cylinders_viz_t(ax, example, t, objs_color='red', robot_color='black')

            ax.set_ylim([-.2, .2])

        anim = FuncAnimation(fig, viz_t, frames=50, interval=2)
        return anim

    @staticmethod
    def propnet_obj_v(batch, batch_size, obj_idx, time, device):
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

    @staticmethod
    def propnet_robot_v(batch, batch_size, time, device):
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

    @staticmethod
    def propnet_add_vel(example: Dict):
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

    @staticmethod
    def propnet_outputs_to_state(inputs, pred_vel, pred_pos, device, obj_dz=0):
        pred_states = {}
        pred_states['height'] = inputs['height']
        pred_states['radius'] = inputs['radius']
        pred_states['num_objs'] = inputs['num_objs']

        pred_robot_pos_3d = homogeneous(pred_pos[:, 0])  # just needs to be 3d, doesn't matter what Z values is
        pred_robot_vel_3d = homogeneous(pred_vel[:, 0])
        pred_states[f'{ARM_HAND_NAME}/tcp_pos'] = torch.unsqueeze(pred_robot_pos_3d, 1).detach()
        pred_states[f'{ARM_HAND_NAME}/tcp_vel'] = torch.unsqueeze(pred_robot_vel_3d, 1).detach()

        for j in range(int(inputs['num_objs'][0])):
            pred_pos_2d = pred_pos[:, j + 1]
            pred_pos_z = torch.tensor(inputs['height'] / 2 + obj_dz, device=device)
            pred_pos_3d = torch.cat([pred_pos_2d, pred_pos_z], -1)
            pred_vel_3d = homogeneous(pred_vel[:, j + 1])
            pred_states[f'obj{j}/position'] = torch.unsqueeze(pred_pos_3d, 1).detach()
            pred_states[f'obj{j}/linear_velocity'] = torch.unsqueeze(pred_vel_3d, 1).detach()

        return pred_states

    @staticmethod
    def propnet_rel(obj_pos, num_objects, relation_dim, is_close_threshold, device=None):
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

    @staticmethod
    def initial_identity_aug_params(batch_size, k_transforms, device):
        return torch.zeros([batch_size, k_transforms, 3], device=device)  # change in x, y, theta (about z)

    @staticmethod
    def sample_target_aug_params(rng: np.random.RandomState, aug_params, n_samples, device):
        trans_lim = np.ones([2]) * aug_params['target_trans_lim']
        trans_target = torch.tensor(rng.uniform(low=-trans_lim, high=trans_lim, size=[n_samples, 2]), device=device,
                                    dtype=torch.float32)

        theta_lim = np.ones([1]) * aug_params['target_euler_lim']
        theta_target = torch.tensor(rng.uniform(low=-theta_lim, high=theta_lim, size=[n_samples, 1]), device=device,
                                    dtype=torch.float32)

        target_params = torch.cat([trans_target, theta_target], -1)
        return target_params

    @staticmethod
    def aug_target_pos(target):
        return torch.cat([target[0], target[1], 0], 0)

    @staticmethod
    def transformation_params_to_matrices(obj_transforms, device):
        xy = obj_transforms[..., :2]
        theta = obj_transforms[..., 2:3]
        zrp = torch.zeros(obj_transforms.shape[:-1] + (3,), device=device)
        xyzrpy = torch.cat([xy, zrp, theta], -1)
        return xyzrpy_to_matrices(xyzrpy)

    def aug_apply_no_ik(self,
                        moved_mask,
                        m,
                        to_local_frame,
                        inputs: Dict,
                        batch_size,
                        device,
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
        zeros_expanded2 = torch.zeros([batch_size, 1, 1, 3], device=device)
        m_expanded = m[:, None]
        no_translation_mask = torch.ones(m_expanded.shape, device=device)
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

    @staticmethod
    def aug_ik(inputs: Dict,
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

    @staticmethod
    def aug_transformation_jacobian(obj_transforms, device):
        """

        Args:
            obj_transforms: [b, k_transforms, p]

        Returns: [b, k_transforms, p, 4, 4]

        """
        zrp = torch.zeros(obj_transforms.shape[:-1] + (3,), device=device)
        xy = obj_transforms[..., :2]
        theta = obj_transforms[..., 2:3]
        xyzrpy = torch.cat([xy, zrp, theta], -1)
        jacobian = transformation_jacobian(xyzrpy)
        jacobian_xy = jacobian[..., 0:2, :, :]
        jacobian_theta = jacobian[..., 2:3, :, :]
        jacobian_xyt = torch.cat([jacobian_xy, jacobian_theta], -3)
        return jacobian_xyt

    @staticmethod
    def aug_distance(transforms1, transforms2):
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
