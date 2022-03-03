from typing import Dict

import torch

from moonshine.grid_utils_torch import batch_center_res_shape_to_origin_point, batch_align_to_grid, batch_point_to_idx, \
    occupied_voxels_to_points_batched, min_max_extent_batched


def create_env_indices(local_env_h_rows: int, local_env_w_cols: int, local_env_c_channels: int, batch_size: int):
    # Construct a [b, h, w, c, 3] grid of the indices which make up the local environment
    pixel_row_indices = torch.arange(0, local_env_h_rows, dtype=torch.float32)
    pixel_col_indices = torch.arange(0, local_env_w_cols, dtype=torch.float32)
    pixel_channel_indices = torch.arange(0, local_env_c_channels, dtype=torch.float32)
    x_indices, y_indices, z_indices = torch.meshgrid(pixel_col_indices, pixel_row_indices, pixel_channel_indices,
                                                     indexing='xy')

    # Make batched versions for creating the local environment
    batch_y_indices = torch.tile(y_indices.unsqueeze(0), [batch_size, 1, 1, 1]).long()
    batch_x_indices = torch.tile(x_indices.unsqueeze(0), [batch_size, 1, 1, 1]).long()
    batch_z_indices = torch.tile(z_indices.unsqueeze(0), [batch_size, 1, 1, 1]).long()

    # Convert for rastering state
    pixel_indices = torch.stack([y_indices, x_indices, z_indices], 3)
    pixel_indices = pixel_indices.unsqueeze(0)
    pixel_indices = torch.tile(pixel_indices, [batch_size, 1, 1, 1, 1])

    return {
        'x':             batch_x_indices,
        'y':             batch_y_indices,
        'z':             batch_z_indices,
        'pixel_indices': pixel_indices
    }


def get_local_env_and_origin_point(center_point,
                                   environment: Dict,
                                   h: int,
                                   w: int,
                                   c: int,
                                   indices: Dict,
                                   batch_size: int):
    res = environment['res']
    full_env_origin_point = environment['origin_point']
    full_env = environment['env']

    local_env_origin_point = batch_center_res_shape_to_origin_point(center=center_point,
                                                                    res=res,
                                                                    h=h,
                                                                    w=w,
                                                                    c=c)

    local_env_origin_point = batch_align_to_grid(local_env_origin_point, full_env_origin_point, res)

    batch_indices, full_env_points = occupied_voxels_to_points_batched(full_env, res, full_env_origin_point)
    local_min_points, local_max_points = min_max_extent_batched(h, w, c, res, local_env_origin_point)  # [b, 3], [b, 3]
    local_min_points_gathered = torch.take_along_dim(local_min_points, batch_indices.unsqueeze(-1), 0)  # [p, 3]
    local_max_points_gathered = torch.take_along_dim(local_max_points, batch_indices.unsqueeze(-1), 0)
    in_bounds = torch.logical_and(local_min_points_gathered < full_env_points,
                                  full_env_points < local_max_points_gathered)  # [p, 3]
    in_bounds = torch.all(in_bounds, -1)  # [p]
    full_env_points_in_bounds_indices = torch.where(in_bounds)[0]  # [k]
    batch_indices_in_bounds = torch.take_along_dim(batch_indices, full_env_points_in_bounds_indices, 0)
    full_env_points_in_bounds = torch.take_along_dim(full_env_points, full_env_points_in_bounds_indices.unsqueeze(-1), 0)

    res_gathered = torch.gather(res, batch_indices_in_bounds, 0)
    ocal_env_origin_point_gathered = torch
    local_env_indices = batch_point_to_idx(full_env_points_in_bounds, res_gathered, local_env_origin_point_gathered)
    batch_indices, h_indices, w_indices, c_indices = torch.unbind(local_env_indices, -1)
    local_env = torch.zeros([batch_size, h, w, c], device=full_env.device)
    local_env[batch_indices, h_indices, w_indices, c_indices] = 1

    return local_env, local_env_origin_point
