import torch

from moonshine.grid_utils_torch import batch_point_to_idx


def points_to_voxel_grid_res_origin_point_batched(batch_indices, points, res, origin_point, h, w, c, batch_size):
    """
    Args:
        batch_indices: [n], batch_indices[i] is the batch indices for point points[i]. Must be int64 type
        points: [n, 3]
        res: [n]
        origin_point: [n, 3]
        h:
        w:
        c:
        batch_size:

    Returns: 1-channel binary voxel grid of shape [b,h,w,c]
    """
    indices = batch_point_to_idx(points, res, origin_point)  # [n, 4]
    batch_indices, h_indices, w_indices, c_indices = indices.unbind(-1)
    voxel_grid = torch.zeros([batch_size, h, w, c], device=points.device)
    voxel_grid[batch_indices, h_indices, w_indices, c_indices] = 1
    return voxel_grid
