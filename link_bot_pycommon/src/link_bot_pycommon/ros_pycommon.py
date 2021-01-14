import numpy as np

import ros_numpy
import rospy
from arc_utilities.listener import Listener
from arc_utilities.tf2wrapper import TF2Wrapper
from link_bot_pycommon import grid_utils
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.grid_utils import extent_to_center, extent_to_env_shape
from peter_msgs.srv import ComputeOccupancyRequest, Position3DEnable, GetPosition3D, Position3DAction
from rosgraph.names import ns_join
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_srvs.srv import Empty
from tf2_sensor_msgs import tf2_sensor_msgs


def get_occupancy(service_provider,
                  env_w_cols,
                  env_h_rows,
                  env_c_channels,
                  res,
                  center_x,
                  center_y,
                  center_z,
                  excluded_models):
    request = ComputeOccupancyRequest()
    request.resolution = res
    request.h_rows = env_h_rows
    request.w_cols = env_w_cols
    request.c_channels = env_c_channels
    request.center.x = center_x
    request.center.y = center_y
    request.center.z = center_z
    request.excluded_models = excluded_models
    request.request_new = True
    # from time import perf_counter
    # t0 = perf_counter()
    response = service_provider.compute_occupancy(request)
    # print('time to compute occupancy', perf_counter() - t0)
    grid = np.array(response.grid).reshape([env_w_cols, env_h_rows, env_c_channels])
    # NOTE: this makes it so we can index with row (y), col (x), channel (z)
    grid = np.transpose(grid, [1, 0, 2])
    return grid, response


def get_environment_for_extents_3d(extent,
                                   res: float,
                                   service_provider: BaseServices,
                                   excluded_models: [str]):
    cx, cy, cz = extent_to_center(extent)
    env_h_rows, env_w_cols, env_c_channels = extent_to_env_shape(extent, res)
    grid, _ = get_occupancy(service_provider,
                            env_w_cols=env_w_cols,
                            env_h_rows=env_h_rows,
                            env_c_channels=env_c_channels,
                            res=res,
                            center_x=cx,
                            center_y=cy,
                            center_z=cz,
                            excluded_models=excluded_models)
    x_min = extent[0]
    y_min = extent[2]
    z_min = extent[4]
    origin_row = -y_min / res
    origin_col = -x_min / res
    origin_channel = -z_min / res
    origin = np.array([origin_row, origin_col, origin_channel], np.int32)
    return {
        'env':    grid,
        'res':    res,
        'origin': origin,
        'extent': extent,
    }


def get_occupancy_data(env_h_m: float,
                       env_w_m: float,
                       res: float,
                       service_provider: BaseServices,
                       robot_name: str):
    """
    :param env_h_m:  meters
    :param env_w_m: meters
    :param res: meters
    :param service_provider: from gazebo_utils
    :param robot_name: model name in gazebo
    :return:
    """
    env_h_rows = int(env_h_m / res)
    env_w_cols = int(env_w_m / res)
    env_c_channels = 1
    if robot_name is None:
        raise ValueError("robot name cannot be None")
    if robot_name == "":
        raise ValueError("robot name cannot be empty string")
    grid, response = get_occupancy(service_provider,
                                   env_w_cols=env_w_cols,
                                   env_h_rows=env_h_rows,
                                   env_c_channels=env_c_channels,
                                   res=res,
                                   center_x=0,
                                   center_y=0,
                                   center_z=res,
                                   # we want to do a little off the ground because grid cells are centered
                                   excluded_models=[robot_name])
    origin = np.array(response.origin)
    full_env_data = grid_utils.OccupancyData(data=grid, resolution=res, origin=origin)
    return full_env_data


def make_movable_object_services(object_name):
    return {
        'enable':       rospy.ServiceProxy(f'{object_name}/enable', Position3DEnable),
        'get_position': rospy.ServiceProxy(f'{object_name}/get', GetPosition3D),
        'set':          rospy.ServiceProxy(f'{object_name}/set', Position3DAction),
        'move':         rospy.ServiceProxy(f'{object_name}/move', Position3DAction),
        'stop':         rospy.ServiceProxy(f'{object_name}/stop', Empty),
    }


def publish_color_image(pub: rospy.Publisher, x):
    color = x.astype(np.uint8)
    color_viz_msg = ros_numpy.msgify(Image, color, encoding="rgb8")
    pub.publish(color_viz_msg)


def publish_depth_image(pub: rospy.Publisher, x):
    depth_viz_msg = ros_numpy.msgify(Image, x, encoding="32FC1")
    pub.publish(depth_viz_msg)


def get_camera_params(camera_name: str):
    camera_params_topic_name = ns_join(ns_join(camera_name, 'qhd'), "camera_info")
    camera_params_listener = Listener(camera_params_topic_name, CameraInfo)
    camera_params: CameraInfo = camera_params_listener.get()
    return camera_params


def transform_points_to_robot_frame(tf: TF2Wrapper, cdcpd_msg: PointCloud2, robot_frame_id: str = 'robot_root'):
    """ transform into robot-frame """
    transform = tf.get_transform_msg(robot_frame_id, cdcpd_msg.header.frame_id)
    cdcpd_points_robot_frame = tf2_sensor_msgs.do_transform_cloud(cdcpd_msg, transform)
    cdcpd_points_array = ros_numpy.numpify(cdcpd_points_robot_frame)
    x = cdcpd_points_array['x']
    y = cdcpd_points_array['y']
    z = cdcpd_points_array['z']
    points = np.stack([x, y, z], axis=-1)
    return points
