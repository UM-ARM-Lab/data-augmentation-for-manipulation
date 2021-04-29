from typing import Dict, Optional

import numpy as np
import tensorflow as tf

import ros_numpy
import rospy
from geometry_msgs.msg import TransformStamped
from mps_shape_completion_msgs.msg import OccupancyStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import MultiArrayDimension, Float32MultiArray


def indeces_to_point(rowcols, resolution, origin):
    return (rowcols - origin) * resolution


def batch_idx_to_point_3d_in_env_tf(row,
                                    col,
                                    channel,
                                    env: Dict):
    origin = tf.cast(env['origin'], tf.int64)
    y = tf.cast(row - tf.gather(origin, 0, axis=-1), tf.float32) * env['res']
    x = tf.cast(col - tf.gather(origin, 1, axis=-1), tf.float32) * env['res']
    z = tf.cast(channel - tf.gather(origin, 2, axis=-1), tf.float32) * env['res']
    return tf.stack([x, y, z], axis=-1)


def idx_to_point_3d_in_env(row: int,
                           col: int,
                           channel: int,
                           env: Dict):
    return idx_to_point_3d_from_extent(row=row, col=col, channel=channel, resolution=env['res'], extent=env['extent'])


def idx_to_point_3d_from_extent(row, col, channel, resolution, extent):
    origin_point = extent_to_origin_point(extent=extent, res=resolution)
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


def extent_to_center(extent_3d):
    min_x, max_x, min_y, max_y, min_z, max_z = extent_3d
    cx = (max_x + min_x) / 2
    cy = (max_y + min_y) / 2
    cz = (max_z + min_z) / 2
    return cx, cy, cz


def extent_to_origin_point(extent, res):
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


def environment_to_occupancy_msg(environment: Dict, frame: str = 'occupancy', stamp=None):
    if stamp is None:
        stamp = rospy.Time.now()

    occupancy = Float32MultiArray()
    env = environment['env']
    # NOTE: The plugin assumes data is ordered [x,y,z] so transpose here
    env = np.transpose(env, [1, 0, 2])
    occupancy.data = env.astype(np.float32).flatten().tolist()
    x_shape, y_shape, z_shape = env.shape
    occupancy.layout.dim.append(MultiArrayDimension(label='x', size=x_shape, stride=x_shape * y_shape * z_shape))
    occupancy.layout.dim.append(MultiArrayDimension(label='y', size=y_shape, stride=y_shape * z_shape))
    occupancy.layout.dim.append(MultiArrayDimension(label='z', size=z_shape, stride=z_shape))
    msg = OccupancyStamped()
    msg.occupancy = occupancy
    msg.scale = environment['res']
    msg.header.stamp = stamp
    msg.header.frame_id = frame
    return msg


def send_occupancy_tf(broadcaster, environment: Dict, frame: str = 'occupancy'):
    transform = TransformStamped()
    transform.header.stamp = rospy.Time.now()
    transform.header.frame_id = "world"
    transform.child_frame_id = frame

    origin_point = extent_to_origin_point(extent=environment['extent'], res=environment['res'])
    origin_x, origin_y, origin_z = origin_point
    transform.transform.translation.x = origin_x
    transform.transform.translation.y = origin_y
    transform.transform.translation.z = origin_z
    transform.transform.rotation.x = 0
    transform.transform.rotation.y = 0
    transform.transform.rotation.z = 0
    transform.transform.rotation.w = 1
    broadcaster.sendTransform(transform)


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
    col = tf.cast(x / resolution + origin[1], tf.int64)
    row = tf.cast(y / resolution + origin[0], tf.int64)
    return row, col


def batch_point_to_idx_tf_3d_in_batched_envs(points, env: Dict):
    x = tf.gather(points, 0, axis=-1)
    y = tf.gather(points, 1, axis=-1)
    z = tf.gather(points, 2, axis=-1)
    col = tf.cast(x / env['res'] + tf.gather(env['origin'], 1, axis=-1), tf.int64)
    row = tf.cast(y / env['res'] + tf.gather(env['origin'], 0, axis=-1), tf.int64)
    channel = tf.cast(z / env['res'] + tf.gather(env['origin'], 2, axis=-1), tf.int64)
    return row, col, channel


def batch_point_to_idx_tf_3d(x,
                             y,
                             z,
                             resolution: float,
                             origin):
    col = tf.cast(x / resolution + origin[1], tf.int64)
    row = tf.cast(y / resolution + origin[0], tf.int64)
    channel = tf.cast(z / resolution + origin[2], tf.int64)
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
