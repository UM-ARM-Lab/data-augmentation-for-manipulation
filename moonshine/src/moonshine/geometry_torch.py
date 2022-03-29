from math import inf

import torch


def homogeneous(points):
    return torch.cat([points, torch.ones_like(points[..., 0:1])], axis=-1)


def transform_points_3d(transform_matrix, points):
    """

    Args:
        transform_matrix: [b1, b2, ..., 4, 4]
        points: [b1, b2, ..., 3]

    Returns:

    """
    points_homo = homogeneous(points)
    points_homo = points_homo.unsqueeze(-1)
    transformed_points = torch.matmul(transform_matrix, points_homo)
    return transformed_points.squeeze(-1)[..., :3]


def rotate_points_3d(rotation_matrix, points):
    """

    Args:
        rotation_matrix: [b1, b2, ..., b2, 3, 3]
        points: [b1, b2, ..., b2, 3]

    Returns:

    """
    rotated_points = torch.matmul(rotation_matrix, points.unsqueeze(-1))
    return rotated_points.squeeze(-1)


def densify_points(batch_size, points, num_densify=5):
    """
    Args:
        points: [b, n, 3]
    Returns: [b, n * num_density, 3]
    """
    if points.shape[1] <= 1:
        return points

    starts = points[:, :-1]
    ends = points[:, 1:]
    m = points.shape[1] - 1
    linspaced = torch.linspace(0, 1, num_densify).to(points.device)  # [num_density]
    linspaced = torch.tile(linspaced[None, None, :, None], [batch_size, m, 1, 3])  # [b, m, num_densify, 3]
    dense_points = starts.unsqueeze(2) + linspaced * (ends - starts).unsqueeze(2)  # [b, m, num_densify, 3
    dense_points = dense_points.reshape([batch_size, -1, 3])
    return dense_points