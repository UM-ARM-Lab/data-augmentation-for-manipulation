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
