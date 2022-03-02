#!/usr/bin/env python

import numpy as np

import rospy
import tf2_ros
from moonshine import grid_utils_tf
from link_bot_pycommon.grid_utils_np import environment_to_vg_msg
from moonshine.get_local_environment_tf import get_local_env_and_origin_3d_tf_old
from moonshine.gpu_config import limit_gpu_mem
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped

limit_gpu_mem(1.0)

res = [0.01]
full_h_rows = 100
full_w_cols = 100
full_c_channels = 50
local_h_rows = 50
local_w_cols = 50
local_c_channels = 50

center_point = np.array([[-.25, -.25, 0.25]], np.float32)

full_env = np.zeros([1, full_h_rows, full_w_cols, full_c_channels], dtype=np.float32)
full_env_origin = np.array([[full_h_rows / 2, full_w_cols / 2, 0]], dtype=np.float32)
# full_env[:, 0:20, 0:20, 0:20] = 1.0
# full_env[:, 50:91, 60:91, 20:50] = 1.0

for _ in range(500):
    row = int(np.random.uniform(0, full_h_rows))
    col = int(np.random.uniform(0, full_w_cols))
    channel = int(np.random.uniform(0, full_c_channels))
    full_env[:, row, col, channel] = 1.0

local_env, local_env_origin = get_local_env_and_origin_3d_tf_old(center_point,
                                                                 full_env,
                                                                 full_env_origin,
                                                                 res,
                                                                 local_h_rows,
                                                                 local_w_cols,
                                                                 local_c_channels)

full_environment = {
    'env': full_env[0],
    'res': res[0],
    'origin': full_env_origin[0],
}

local_environment = {
    'env': local_env[0],
    'res': res[0],
    'origin': local_env_origin[0],
}

rospy.init_node("test_get_local_env")
broadcaster = tf2_ros.StaticTransformBroadcaster()
local_occupancy_pub = rospy.Publisher('local_occupancy', VoxelgridStamped, queue_size=10, latch=True)
full_pub = rospy.Publisher('occupancy', VoxelgridStamped, queue_size=10, latch=True)

local_occupancy_msg = environment_to_vg_msg(local_environment, frame='local_occupancy')
full_occupancy_msg = environment_to_vg_msg(full_environment, frame='occupancy')

while True:
    rospy.sleep(0.1)
    grid_utils_tf.send_voxelgrid_tf(broadcaster, local_environment, frame='local_occupancy')
    local_occupancy_pub.publish(local_occupancy_msg)

    grid_utils_tf.send_voxelgrid_tf(broadcaster, full_environment, frame='occupancy')
    full_pub.publish(full_occupancy_msg)
