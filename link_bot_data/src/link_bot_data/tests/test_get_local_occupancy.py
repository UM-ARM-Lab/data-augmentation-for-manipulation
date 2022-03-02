#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import rospy

from link_bot_gazebo import gazebo_services
from link_bot_pycommon.grid_utils_np import compute_extent
from link_bot_pycommon.get_occupancy import get_occupancy_data
from moonshine.get_local_environment_tf import get_local_env_and_origin

rospy.init_node("testing")

service_provider = gazebo_services.GazeboServices()

h = 2
w = 2
res = 0.01

# get full env once
full_env_data = get_occupancy_data(env_w_m=w,
                                   env_h_m=h,
                                   res=res,
                                   service_provider=service_provider,
                                   robot_name='car')

local_env_center = np.array([0.45, 0.0])
local_h_rows = 100
local_w_cols = 100

local_env, local_env_origin = get_local_env_and_origin(center_point=local_env_center,
                                                       full_env=full_env_data.data,
                                                       full_env_origin=full_env_data.origin,
                                                       res=res,
                                                       local_h_rows=local_h_rows,
                                                       local_w_cols=local_w_cols)
local_env_extent = compute_extent(local_h_rows, local_w_cols, res, local_env_origin)

plt.figure()
plt.imshow(np.flipud(local_env), extent=local_env_extent)
ax = plt.gca()
ax.set_xticks(np.arange(local_env_extent[0], local_env_extent[1], res))
ax.set_yticks(np.arange(local_env_extent[2], local_env_extent[3], res))
ax.set_xticklabels([])
ax.set_yticklabels([])
# plt.grid()
plt.show()
