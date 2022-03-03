from typing import Dict

import torch

from moonshine.grid_utils_torch import batch_center_res_shape_to_origin_point, batch_align_to_grid, round_to_res


def swap_xy(x):
    """

    Args:
        x: has shape [b1, b2, ..., bn, 3]
        n_batch_dims: same as n in the above shape, number of dimensions before the dimension of 3 (x,y,z)

    Returns: the x/y will be swapped

    """
    first = x[..., 0]
    second = x[..., 1]
    z = x[..., 2]
    swapped = torch.stack([second, first, z], dim=-1)
    return swapped


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

    res_expanded = res.unsqueeze(-1)
    local_to_full_offset_xyz = round_to_res(local_env_origin_point - full_env_origin_point, res_expanded)
    local_to_full_offset = swap_xy(local_to_full_offset_xyz)

    # Transform into coordinate of the full_env
    tile_sizes = [1, h, w, c]

    batch_y_indices_in_full_env_frame = indices['y'] + local_to_full_offset[:, 0, None, None, None]
    batch_x_indices_in_full_env_frame = indices['x'] + local_to_full_offset[:, 1, None, None, None]
    batch_z_indices_in_full_env_frame = indices['z'] + local_to_full_offset[:, 2, None, None, None]

    batch_indices = torch.arange(0, batch_size, dtype=torch.int64).to(center_point.device)
    batch_indices = batch_indices[:, None, None, None]
    batch_indices = torch.tile(batch_indices, tile_sizes)
    local_env = full_env[batch_indices,
                         batch_y_indices_in_full_env_frame,
                         batch_x_indices_in_full_env_frame,
                         batch_z_indices_in_full_env_frame]

    return local_env, local_env_origin_point
