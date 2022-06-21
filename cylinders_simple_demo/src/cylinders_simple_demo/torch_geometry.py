import torch


def batch_center_res_shape_to_origin_point(center, res, h, w, c):
    shape_xyz = torch.tensor([w, h, c], dtype=torch.float32).to(center.device)
    return center - (shape_xyz * torch.unsqueeze(res, -1) / 2)


def batch_align_to_grid(point, origin_point, res):
    """

    Args:
        point: [n, 3], meters, in the same frame as origin_point
        origin_point: [n, 3], meters, in the same frame as point
        res: [n], meters

    Returns:

    """
    res_expanded = torch.unsqueeze(res, -1)
    return (torch.round((point - origin_point) / res_expanded)).float() * res_expanded + origin_point


def round_to_res(x, res):
    # helps with stupid numerics issues
    return torch.round(x / res).long()


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


def batch_point_to_idx(points, res, origin_point):
    """

    Args:
        points: [b1,...,bn,3] points in a frame, call it world
        res: [b1,...,b2] meters
        origin_point: [b1,...,b2,3] the position [x,y,z] of the center of the voxel (0,0,0) in the same frame as points

    Returns:

    """
    return swap_xy(round_to_res((points - origin_point), res.unsqueeze(-1)))


def batch_idx_to_point_3d_res_origin_point(indices, res, origin_point):
    indices_xyz = swap_xy(indices)
    return indices_xyz.float() * res.unsqueeze(-1) + origin_point


def occupied_voxels_to_points_batched(vg, res, origin_point):
    all_indices = torch.where(vg > 0.5)
    batch_indices = all_indices[0]
    indices = torch.stack(all_indices[1:], -1)
    res_gathered = torch.take_along_dim(res, batch_indices, 0)
    origin_point_gathered = torch.take_along_dim(origin_point, batch_indices.unsqueeze(-1), 0)
    occupied_points = batch_idx_to_point_3d_res_origin_point(indices, res_gathered, origin_point_gathered)
    return batch_indices, occupied_points


def min_max_extent_batched(rows: int, cols: int, channels: int, resolution, origin):
    batch_size = resolution.shape[0]
    min_idx = torch.tile(torch.zeros(3, device=resolution.device).unsqueeze(0), [batch_size, 1])
    max_idx = torch.tile(torch.tensor([rows, cols, channels], device=resolution.device).unsqueeze(0), [batch_size, 1])
    min_points = batch_idx_to_point_3d_res_origin_point(min_idx, resolution, origin)
    max_points = batch_idx_to_point_3d_res_origin_point(max_idx, resolution, origin)
    return min_points, max_points
