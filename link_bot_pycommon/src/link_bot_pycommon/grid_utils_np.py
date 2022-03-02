from typing import Dict, Optional

import numpy as np

import ros_numpy
import rospy
from rviz_voxelgrid_visuals import conversions
from sensor_msgs.msg import PointCloud2


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
