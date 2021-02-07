import numpy as np

from link_bot_pycommon import grid_utils
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.grid_utils import extent_to_center, extent_to_env_shape
from peter_msgs.srv import ComputeOccupancyRequest


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