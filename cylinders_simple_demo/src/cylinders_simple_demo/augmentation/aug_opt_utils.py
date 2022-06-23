from dataclasses import dataclass

import numpy as np
import torch

from cylinders_simple_demo.utils.torch_geometry import transform_points_3d


@dataclass
class MinDists:
    attract: torch.Tensor
    repel: torch.Tensor
    robot_repel: torch.Tensor


@dataclass
class EnvPoints:
    full: torch.Tensor
    sparse: torch.Tensor


def subsample_points(points, fraction):
    """

    Args:
        points: [n, 3]
        fraction: from 0.0 to 1.0

    Returns:

    """
    n_take_every = int(1 / fraction)
    return points[::n_take_every]


def transform_obj_points(obj_points, moved_mask, transformation_matrices):
    """

    Args:
        obj_points: [b,m,T,n_points,3]
        moved_mask: [b,m,...]
        transformation_matrices: [b,k,4,4]
            considered to in the frame of the obj_points,
            which is defined as the same orientation as the world but with the position being the center of obj_points

    Returns: [b,k,T,n_points,3], [b,3]

    """
    to_local_frame_moved_mean = get_local_frame(moved_mask, obj_points)
    to_local_frame_moved_mean_expanded = to_local_frame_moved_mean[:, None, None, None, :]

    obj_points_local_frame = obj_points - to_local_frame_moved_mean_expanded  # [b, m_objects, T, n_points, 3]
    transformation_matrices_expanded = transformation_matrices[..., None, None, :, :, :]
    obj_points_aug_local_frame = transform_points_3d(transformation_matrices_expanded, obj_points_local_frame)
    obj_points_aug = obj_points_aug_local_frame + to_local_frame_moved_mean_expanded  # [b, m_objects, T, n_points, 3]
    return obj_points_aug, to_local_frame_moved_mean


def get_local_frame(moved_mask, obj_points):
    to_local_frame = obj_points.mean(-2)  # [b,m,T,3]
    to_local_frame = to_local_frame.mean(-2)  # [b,m,3]
    to_local_frame_moved_mean = mean_over_moved(moved_mask, to_local_frame)  # [b, 3]
    return to_local_frame_moved_mean


def sum_over_moved(moved_mask, x):
    """
    Gives the sum of x over dimension 1, which represent different objects, but the non-moved objects aren't included.

    Args:
        moved_mask: [b, m]
        x: [b, m, d1, ..., dn]

    Returns:

    """
    # replacing the values where moved_mask is false with zero will not affect the sum...
    moved_mask_expanded = expand_to_match(moved_mask, x)
    x_moved = torch.where(moved_mask_expanded.bool(), x, torch.zeros_like(x))
    x_moved_sum = x_moved.sum(1)
    return x_moved_sum


def mean_over_moved(moved_mask, x):
    """
    Gives the mean of x over dimension 1, which represent different objects, but the non-moved objects aren't included.

    Args:
        moved_mask: [b, m]
        x: [b, m, d1, ..., dn]

    Returns:

    """
    # replacing the values where moved_mask is false with zero will not affect the sum...
    x_moved_sum = sum_over_moved(moved_mask, x)
    # ... if we divide by the right numbers
    moved_count = moved_mask.sum(1)
    moved_count = expand_to_match(moved_count, x)
    x_moved_mean = x_moved_sum / moved_count
    return x_moved_mean


def expand_to_match(a, b):
    """

    Args:
        a: [b1, b2, ..., bN]
        b: [b1, b2, ..., bN, d1, d2, ... dN]

    Returns: a but with the shape [b1, b2, ..., bN, 1, 1, ... 1]

    """
    a_expanded = a
    for dim_j in range(b.ndim - 2):
        a_expanded = torch.unsqueeze(a_expanded, dim=-1)
    return a_expanded


def check_env_constraints(attract_mask, min_dist):
    # NOTE: SDF can be exactly 0 if OOB lookup was done.
    #  We want OOB to count as free-space, so repel satisfied should be 1 if min_dist == 0
    #  attract satisfied should be 0 if min_dist == 0
    attract_satisfied = (min_dist < 0).float()
    repel_satisfied = (min_dist >= 0).float()
    constraints_satisfied = (attract_mask * attract_satisfied) + ((1 - attract_mask) * repel_satisfied)
    return constraints_satisfied


def pick_best_params(invariance_model, sampled_params, batch_size):
    predicted_errors = invariance_model.evaluate(sampled_params)
    _, best_indices_all = torch.topk(-predicted_errors, batch_size, sorted=False)
    rng = np.random.RandomState(0)
    shuffle_indices = rng.permutation(range(batch_size))
    best_indices_shuffled = best_indices_all[shuffle_indices]
    best_indices = best_indices_shuffled[:batch_size]
    best_params = sampled_params[best_indices]
    return best_params


def delta_min_dist_loss(sdf_dist, sdf_dist_aug):
    min_dist = torch.min(sdf_dist, dim=1)[0]
    min_dist_aug = torch.min(sdf_dist_aug, dim=1)[0]
    delta_min_dist = torch.abs(min_dist - min_dist_aug)
    return delta_min_dist


def dpoint_to_dparams(dpoint, dpoint_dparams):
    """

    Args:
        dpoint:  [b,m,T,n_points,3]
        dpoint_dparams: [b,m,T,n_points,3,p]

    Returns: [b,m,T,n_points,p]

    """
    return torch.squeeze(torch.matmul(torch.unsqueeze(dpoint, -2), dpoint_dparams), dim=-2)
