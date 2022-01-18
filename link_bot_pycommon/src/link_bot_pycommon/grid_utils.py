from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from deprecated import deprecated

import ros_numpy
import rospy
from moonshine.moonshine_utils import swap_xy
from moonshine.numpify import numpify
from rviz_voxelgrid_visuals import conversions
from sensor_msgs.msg import PointCloud2


def binary_or(a, b):
    return tf.clip_by_value(a + b, 0, 1)


def binary_and(a, b):
    return a * b


def subtract(a, b):
    return tf.clip_by_value(a - b, 0, 1)


def lookup_points_in_vg(state_points, env, res, origin_point, batch_size):
    """
    Returns the values of env at state_points
    Args:
        state_points: [b, n, 3], in same frame as origin_point
        env: [b, h, w, c]
        res:
        origin_point: [b, 3] in same frame as state_points
        batch_size:

    Returns: [b, n]

    """
    n_points = state_points.shape[1]
    vg_indices = batch_point_to_idx(state_points,
                                    tf.expand_dims(res, axis=1),
                                    tf.expand_dims(origin_point, axis=1))
    batch_indices = tf.tile(tf.range(batch_size, dtype=tf.int64)[:, None, None], [1, n_points, 1])
    batch_and_vg_indices = tf.concat([batch_indices, vg_indices], axis=-1)
    occupancy_at_state_points = tf.gather_nd(env, batch_and_vg_indices)
    return occupancy_at_state_points


def occupied_voxels_to_points(vg, res, origin_point):
    indices = tf.where(vg > 0.5)
    occupied_points = batch_idx_to_point_3d_tf_res_origin_point(indices, res, origin_point)
    return occupied_points


def occupied_voxels_to_points_batched(vg, res, origin_point):
    all_indices = tf.where(vg > 0.5)
    batch_indices = all_indices[:, 0]
    indices = all_indices[:, 1:]
    res_gathered = tf.gather(res, batch_indices, axis=0)
    origin_point_gathered = tf.gather(origin_point, batch_indices, axis=0)
    occupied_points = batch_idx_to_point_3d_tf_res_origin_point(indices, res_gathered, origin_point_gathered)
    return batch_indices, occupied_points


def round_to_res(x, res):
    # helps with stupid numerics issues
    return tf.cast(tf.round(x / res), tf.int64)


def batch_align_to_grid_tf(point, origin_point, res):
    """

    Args:
        point: [n, 3], meters, in the same frame as origin_point
        origin_point: [n, 3], meters, in the same frame as point
        res: [n], meters

    Returns:

    """
    res_expanded = tf.expand_dims(res, axis=-1)
    return tf.cast(tf.round((point - origin_point) / res_expanded), tf.float32) * res_expanded + origin_point


def voxel_grid_distance(a, b):
    return tf.abs(a - b)


def indeces_to_point(rowcols, resolution, origin):
    return (rowcols - origin) * resolution


def batch_idx_to_point_3d_tf_res_origin_point(indices, res, origin_point):
    indices_xyz = swap_xy(indices)
    return tf.cast(indices_xyz, tf.float32) * tf.expand_dims(res, axis=-1) + origin_point


def batch_idx_to_point_3d_in_env_tf_res_origin(row, col, channel, res, origin):
    origin = tf.cast(origin, tf.int64)
    y = tf.cast(row - tf.gather(origin, 0, axis=-1), tf.float32) * res
    x = tf.cast(col - tf.gather(origin, 1, axis=-1), tf.float32) * res
    z = tf.cast(channel - tf.gather(origin, 2, axis=-1), tf.float32) * res
    return tf.stack([x, y, z], axis=-1)


def batch_idx_to_point_3d_in_env_tf(row, col, channel, env: Dict):
    return batch_idx_to_point_3d_in_env_tf_res_origin(row, col, channel, env['res'], env['origin'])


def idx_to_point_3d_in_env(row: int,
                           col: int,
                           channel: int,
                           env: Dict):
    return idx_to_point_3d_from_extent(row=row, col=col, channel=channel, resolution=env['res'], extent=env['extent'])


def idx_to_point_3d_from_extent(row, col, channel, resolution, extent):
    origin_point = extent_res_to_origin_point(extent=extent, res=resolution)
    y = origin_point[1] + row * resolution
    x = origin_point[0] + col * resolution
    z = origin_point[2] + channel * resolution

    return np.array([x, y, z])


def idx_to_point_3d(row: int,
                    col: int,
                    channel: int,
                    resolution: float,
                    origin: np.ndarray):
    y = (row - origin[0]) * resolution
    x = (col - origin[1]) * resolution
    z = (channel - origin[2]) * resolution
    return np.array([x, y, z])


def idx_to_point_3d_tf(row: int,
                       col: int,
                       channel: int,
                       resolution: float,
                       origin: np.ndarray):
    y = (row - origin[0]) * resolution
    x = (col - origin[1]) * resolution
    z = (channel - origin[2]) * resolution
    return tf.stack([x, y, z], axis=-1)


def idx_to_point(row: int,
                 col: int,
                 resolution: float,
                 origin: np.ndarray):
    y = (row - origin[0]) * resolution
    x = (col - origin[1]) * resolution
    return np.array([x, y])


def center_point_to_origin_indices(h_rows: int,
                                   w_cols: int,
                                   center_x: float,
                                   center_y: float,
                                   res: float):
    env_origin_x = center_x - w_cols / 2 * res
    env_origin_y = center_y - h_rows / 2 * res
    return np.array([int(-env_origin_x / res), int(-env_origin_y / res)])


def compute_extent_3d(rows: int,
                      cols: int,
                      channels: int,
                      resolution: float,
                      origin: Optional = None):
    if origin is None:
        origin = np.array([rows // 2, cols // 2, channels // 2], np.int32)
    xmin, ymin, zmin = idx_to_point_3d(0, 0, 0, resolution, origin)
    xmax, ymax, zmax = idx_to_point_3d(rows, cols, channels, resolution, origin)
    return np.array([xmin, xmax, ymin, ymax, zmin, zmax], dtype=np.float32)


def batch_extent_to_env_size_tf(extent_3d):
    extent_3d = tf.reshape(extent_3d, [-1, 3, 2])
    min = tf.gather(extent_3d, 0, axis=-1)
    max = tf.gather(extent_3d, 1, axis=-1)
    return tf.abs(max - min)


def extent_to_env_size(extent_3d):
    min_x, max_x, min_y, max_y, min_z, max_z = extent_3d
    env_h_m = abs(max_y - min_y)
    env_w_m = abs(max_x - min_x)
    env_c_m = abs(max_z - min_z)
    return env_h_m, env_w_m, env_c_m


def extent_to_env_shape(extent, res):
    extent = np.array(extent).astype(np.float32)
    res = np.float32(res)
    env_h_m, env_w_m, env_c_m = extent_to_env_size(extent)
    env_h_rows = int(env_h_m / res)
    env_w_cols = int(env_w_m / res)
    env_c_channels = int(env_c_m / res)
    return env_h_rows, env_w_cols, env_c_channels


def batch_extent_to_env_shape_xyz_tf(extent, res):
    extent = tf.cast(extent, tf.float32)
    res = tf.cast(res, tf.float32)
    env_size = batch_extent_to_env_size_tf(extent)
    return round_to_res(env_size, tf.expand_dims(res, axis=-1))


def batch_extent_to_center_tf(extent_3d):
    extent_3d = tf.reshape(extent_3d, [-1, 3, 2])
    return tf.reduce_mean(extent_3d, axis=-1)


def extent_to_center(extent_3d):
    min_x, max_x, min_y, max_y, min_z, max_z = extent_3d
    cx = (max_x + min_x) / 2
    cy = (max_y + min_y) / 2
    cz = (max_z + min_z) / 2
    return cx, cy, cz


def batch_center_res_shape_to_origin_point(center, res, h, w, c):
    shape_xyz = tf.stack([w, h, c], axis=-1)
    return center - (tf.cast(shape_xyz, tf.float32) * tf.expand_dims(res, axis=-1) / 2)


@deprecated
def batch_extent_to_origin_point_tf(extent, res):
    center_xyz = batch_extent_to_center_tf(extent_3d=extent)
    shape_xyz = batch_extent_to_env_shape_xyz_tf(extent=extent, res=res)
    return center_xyz - (tf.cast(shape_xyz, tf.float32) * tf.expand_dims(res, axis=-1) / 2)


def extent_res_to_origin_point(extent, res):
    """

    Args:
        extent: [minx, maxx, miny, maxy, minz, maxz]
        res: scalar

    Returns: the origin point is the x, y, z translation
      from the center of the index[0,0,0] voxel to the point [0.0, 0.0, 0.0] in the voxel grid's frame (usually world)

    """
    center = extent_to_center(extent_3d=extent)
    h_rows, w_cols, c_channels = extent_to_env_shape(extent=extent, res=res)
    oy = center[1] - (h_rows * res / 2)
    ox = center[0] - (w_cols * res / 2)
    oz = center[2] - (c_channels * res / 2)
    return np.array([ox, oy, oz])


def environment_to_pc2(environment: Dict, frame_id: str = 'occupancy', stamp=None):
    return voxel_grid_to_pc2(environment['env'], environment['res'], frame_id, stamp)


def voxel_grid_to_pc2(voxel_grid: np.ndarray, scale: float, frame_id: str, stamp: rospy.Time):
    intensity = voxel_grid.flatten()
    slices = [slice(0, s) for s in voxel_grid.shape]
    indices = np.reshape(np.mgrid[slices], [3, -1]).T
    points = indices * scale + scale / 2
    points = list(zip(points[:, 0], points[:, 1], points[:, 2], intensity))
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)]
    np_record_array = np.array(points, dtype=dtype)
    msg = ros_numpy.msgify(PointCloud2, np_record_array, frame_id=frame_id, stamp=stamp)
    return msg


def vox_to_voxelgrid_stamped(env, scale, frame: str = 'vg', stamp=None, color=None):
    # NOTE: The plugin assumes data is ordered [x,y,z] so transpose here
    env = np.transpose(env, [1, 0, 2])
    msg = conversions.vox_to_voxelgrid_stamped(voxel_grid=env, scale=scale, frame_id=frame)

    if stamp is None:
        msg.header.stamp = rospy.Time.now()
    else:
        msg.header.stamp = stamp

    if color is not None:
        msg.has_color = True
        msg.color = color

    return msg


def environment_to_vg_msg(environment: Dict, frame: str = 'vg', stamp=None, color=None):
    return vox_to_voxelgrid_stamped(environment['env'], environment['res'], frame, stamp, color)


def send_voxelgrid_tf(broadcaster, environment: Dict, frame: str = 'vg'):
    send_voxelgrid_tf_extent_res(broadcaster, environment['extent'], environment['res'], frame)


def send_voxelgrid_tf_extent_res(broadcaster, extent, res, frame: str = 'vg'):
    origin_point = extent_res_to_origin_point(extent, res)
    send_voxelgrid_tf_origin_point_res(broadcaster, origin_point, res, frame)


def send_voxelgrid_tf_origin_point_res(broadcaster,
                                       origin_point,
                                       res,
                                       child_frame_id: str = 'vg',
                                       parent_frame_id: str = 'world'):
    origin_xyz = origin_point - (res / 2)
    origin_xyz = numpify(origin_xyz)
    # the rviz plugin displays the boxes with the corner at the given translation, not the center
    # but the origin_point is at the center, so this offsets things correctly
    broadcaster.sendTransform(origin_xyz, [0, 0, 0, 1], rospy.Time.now(), parent=parent_frame_id, child=child_frame_id)


def compute_extent(rows: int,
                   cols: int,
                   resolution: float,
                   origin: np.ndarray):
    """
    :param rows: scalar
    :param cols: scalar
    :param resolution: scalar
    :param origin: [2]
    :return:
    """
    xmin, ymin = idx_to_point(0, 0, resolution, origin)
    xmax, ymax = idx_to_point(rows, cols, resolution, origin)
    return np.array([xmin, xmax, ymin, ymax], dtype=np.float32)


def batch_point_to_idx_tf(x,
                          y,
                          resolution: float,
                          origin):
    col = round_to_res(x, resolution + origin[1])
    row = round_to_res(y, resolution + origin[0])
    return row, col


def batch_point_to_idx(points, res, origin_point):
    """

    Args:
        points: [b,3] points in a frame, call it world
        res: [b] meters
        origin_point: [b,3] the position [x,y,z] of the center of the voxel (0,0,0) in the same frame as points

    Returns:

    """
    return swap_xy(round_to_res((points - origin_point), tf.expand_dims(res, axis=-1)))


def batch_point_to_idx_tf_3d_res_origin(points, res, origin):
    x = tf.gather(points, 0, axis=-1)
    y = tf.gather(points, 1, axis=-1)
    z = tf.gather(points, 2, axis=-1)
    col = tf.cast(x / res + tf.gather(origin, 1, axis=-1), tf.int64)
    row = tf.cast(y / res + tf.gather(origin, 0, axis=-1), tf.int64)
    channel = tf.cast(z / res + tf.gather(origin, 2, axis=-1), tf.int64)
    return row, col, channel


def batch_point_to_idx_tf_3d_in_batched_envs(points, env: Dict):
    return batch_point_to_idx(points, env['res'], env['origin_point'])


def batch_point_to_idx_tf_3d(x,
                             y,
                             z,
                             resolution: float,
                             origin):
    col = round_to_res(x, resolution + origin[1])
    row = round_to_res(y, resolution + origin[0])
    channel = round_to_res(z, resolution + origin[2])
    return row, col, channel


def point_to_idx_3d_in_env(x: float,
                           y: float,
                           z: float,
                           environment: Dict):
    return point_to_idx_3d(x, y, z, resolution=environment['res'], origin=environment['origin'])


def point_to_idx_3d(x: float,
                    y: float,
                    z: float,
                    resolution: float,
                    origin: np.ndarray):
    row = int(y / resolution + origin[0])
    col = int(x / resolution + origin[1])
    channel = int(z / resolution + origin[2])
    return row, col, channel


def point_to_idx(x: float,
                 y: float,
                 resolution: float,
                 origin: np.ndarray):
    col = int(x / resolution + origin[1])
    row = int(y / resolution + origin[0])
    return row, col


class OccupancyData:

    def __init__(self,
                 data: np.ndarray,
                 resolution: float,
                 origin: np.ndarray):
        """

        :param data:
        :param resolution: scalar, assuming square pixels
        :param origin:
        """
        self.data = data.astype(np.float32)
        self.resolution = resolution
        # Origin means the indeces (row/col) of the world point (0, 0)
        self.origin = origin.astype(np.float32)
        self.extent = compute_extent(self.data.shape[0], self.data.shape[1], resolution, origin)
        # NOTE: when displaying an 2d data as an image, matplotlib assumes rows increase going down,
        #  but rows correspond to y which increases going up
        self.image = np.flipud(self.data)

    def copy(self):
        copy = OccupancyData(data=np.copy(self.data),
                             resolution=self.resolution,
                             origin=np.copy(self.origin))
        return copy


def pad_voxel_grid(voxel_grid, origin, res, new_shape):
    assert voxel_grid.shape[0] <= new_shape[0]
    assert voxel_grid.shape[1] <= new_shape[1]
    assert voxel_grid.shape[2] <= new_shape[2]

    h_pad1 = tf.math.floor((new_shape[0] - voxel_grid.shape[0]) / 2)
    h_pad2 = tf.math.ceil((new_shape[0] - voxel_grid.shape[0]) / 2)

    w_pad1 = tf.math.floor((new_shape[1] - voxel_grid.shape[1]) / 2)
    w_pad2 = tf.math.ceil((new_shape[1] - voxel_grid.shape[1]) / 2)

    c_pad1 = tf.math.floor((new_shape[2] - voxel_grid.shape[2]) / 2)
    c_pad2 = tf.math.ceil((new_shape[2] - voxel_grid.shape[2]) / 2)

    padded_env = tf.pad(voxel_grid, paddings=[[h_pad1, h_pad2], [w_pad1, w_pad2], [c_pad1, c_pad2]])
    new_origin = origin + [h_pad1, w_pad1, c_pad1]
    new_extent = compute_extent_3d(new_shape[0], new_shape[1], new_shape[2], res, new_origin)

    return padded_env, new_origin, new_extent


if __name__ == '__main__':
    import rospy
    from arc_utilities.ros_helpers import get_connected_publisher

    rospy.init_node("test_voxel_grid_to_pc2")
    pub = get_connected_publisher('env_aug', PointCloud2, queue_size=10)
    voxel_grid = np.zeros([10, 10, 10], dtype=np.float32)
    voxel_grid[0, 0, 0] = 1
    voxel_grid[-1, -1, -1] = 1
    voxel_grid[0, -1, -1] = 1
    voxel_grid[-1, 0, -1] = 1
    voxel_grid[-1, 0, 0] = 1
    msg = voxel_grid_to_pc2(voxel_grid, 0.01, 'world', rospy.Time.now())
    pub.publish(msg)

    rospy.sleep(1)
