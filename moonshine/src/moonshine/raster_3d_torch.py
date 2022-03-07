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
    indices = batch_point_to_idx(points, res, origin_point)  # [n, 3]
    # zeros = torch.zeros_like(indices)
    in_bounds = torch.logical_and(torch.all(0 < indices, -1),
                                  torch.all(indices < torch.tensor([h, w, c], device=points.device), -1))
    in_bounds_batch_indices, in_bounds_point_indices = torch.where(in_bounds)

    in_bounds_indices = indices[in_bounds_batch_indices, in_bounds_point_indices]
    in_bounds_h_indices, in_bounds_w_indices, in_bounds_c_indices = in_bounds_indices.unbind(-1)

    voxel_grid = torch.zeros([batch_size, h, w, c], device=points.device)
    voxel_grid[in_bounds_batch_indices, in_bounds_h_indices, in_bounds_w_indices, in_bounds_c_indices] = 1
    return voxel_grid
