import torch

from moonshine.grid_utils_torch import batch_point_to_idx


def points_to_voxel_grid_res_origin_point_batched(points, res, origin_point, h, w, c, batch_size):
    """
    Args:
        points: [b,n,3]
        res: [b]
        origin_point: [b,3]
        h:
        w:
        c:
        batch_size:

    Returns: 1-channel binary voxel grid of shape [b1,...,b2,h,w,c]
    """
    n = points.shape[1]
    res = torch.tile(res[:, None], [1, n])
    origin_point = torch.tile(origin_point[:, None, :], [1, n, 1])
    indices = batch_point_to_idx(points, res, origin_point)  # [n, 4]
    batch_indices = torch.arange(0, batch_size, dtype=torch.long, device=points.device)
    batch_indices = torch.tile(batch_indices[:, None], [1, n])
    h_indices, w_indices, c_indices = indices.unbind(-1)
    voxel_grid = torch.zeros([batch_size, h, w, c], device=points.device)
    voxel_grid[batch_indices, h_indices, w_indices, c_indices] = 1
    return voxel_grid
