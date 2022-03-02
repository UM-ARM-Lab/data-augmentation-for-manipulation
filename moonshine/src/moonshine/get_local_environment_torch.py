from typing import Dict

import tensorflow as torch

from moonshine.grid_utils_torch import batch_center_res_shape_to_origin_point, batch_align_to_grid, round_to_res
from moonshine.tensorflow_utils import swap_xy


def create_env_indices(local_env_h_rows: int, local_env_w_cols: int, local_env_c_channels: int, batch_size: int):
    # Construct a [b, h, w, c, 3] grid of the indices which make up the local environment
    pixel_row_indices = torch.arange(0, local_env_h_rows, dtype=torch.float32)
    pixel_col_indices = torch.arange(0, local_env_w_cols, dtype=torch.float32)
    pixel_channel_indices = torch.arange(0, local_env_c_channels, dtype=torch.float32)
    x_indices, y_indices, z_indices = torch.meshgrid(pixel_col_indices, pixel_row_indices, pixel_channel_indices)

    # Make batched versions for creating the local environment
    batch_y_indices = torch.tile(y_indices.unsqueeze(0), [batch_size, 1, 1, 1]).int()
    batch_x_indices = torch.tile(x_indices.unsqueeze(0), [batch_size, 1, 1, 1]).int()
    batch_z_indices = torch.tile(z_indices.unsqueeze(0), [batch_size, 1, 1, 1]).int()

    # Convert for rastering state
    pixel_indices = torch.stack([y_indices, x_indices, z_indices], axis=3)
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

    batch_indices = torch.tile(torch.arange(0, batch_size, dtype=torch.int64)[:, None, None, None], tile_sizes)
    gather_indices = torch.stack([batch_indices,
                                  batch_y_indices_in_full_env_frame,
                                  batch_x_indices_in_full_env_frame,
                                  batch_z_indices_in_full_env_frame],
                                 axis=4)
    local_env = torch.gather_nd(full_env, gather_indices)

    return local_env, local_env_origin_point
