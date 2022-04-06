from typing import Dict
from distutils.version import StrictVersion
import torch.nn.functional as F
import torch

from moonshine.grid_utils_torch import batch_center_res_shape_to_origin_point, batch_align_to_grid, round_to_res, \
    swap_xy


def create_env_indices(local_env_h_rows: int, local_env_w_cols: int, local_env_c_channels: int, batch_size: int):
    # Construct a [b, h, w, c, 3] grid of the indices which make up the local environment
    pixel_row_indices = torch.arange(0, local_env_h_rows, dtype=torch.float32)
    pixel_col_indices = torch.arange(0, local_env_w_cols, dtype=torch.float32)
    pixel_channel_indices = torch.arange(0, local_env_c_channels, dtype=torch.float32)
    if torch.__version__ == '1.9.1+cu102':
        y_indices, x_indices, z_indices = torch.meshgrid(pixel_col_indices, pixel_row_indices, pixel_channel_indices)
    else:
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

    y_min = int(batch_y_indices_in_full_env_frame.min().detach().cpu())
    y_max = int(batch_y_indices_in_full_env_frame.max().detach().cpu())
    x_min = int(batch_x_indices_in_full_env_frame.min().detach().cpu())
    x_max = int(batch_x_indices_in_full_env_frame.max().detach().cpu())
    z_min = int(batch_z_indices_in_full_env_frame.min().detach().cpu())
    z_max = int(batch_z_indices_in_full_env_frame.max().detach().cpu())
    z_begin = max(0 - z_min, 0)
    z_end = max(z_max - (full_env.shape[3] - 1), 0)
    x_begin = max(0 - x_min, 0)
    x_end = max(x_max - (full_env.shape[2] - 1), 0)
    y_begin = max(0 - y_min, 0)
    y_end = max(y_max - (full_env.shape[1] - 1), 0)
    paddings = [z_begin, z_end, x_begin, x_end, y_begin, y_end]
    full_env_padded = F.pad(full_env, paddings)
    batch_indices = torch.arange(0, batch_size, dtype=torch.int64).to(center_point.device)
    batch_indices = batch_indices[:, None, None, None]
    batch_indices = torch.tile(batch_indices, tile_sizes)
    local_env = full_env_padded[batch_indices,
                                batch_y_indices_in_full_env_frame + y_begin,
                                batch_x_indices_in_full_env_frame + x_begin,
                                batch_z_indices_in_full_env_frame + z_begin]

    return local_env, local_env_origin_point
