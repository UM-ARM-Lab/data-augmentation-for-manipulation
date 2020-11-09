#!/usr/bin/env python
from time import sleep

import rospy
from link_bot_gazebo_python import gazebo_services
from link_bot_pycommon import grid_utils
from link_bot_pycommon.ros_pycommon import get_environment_for_extents_3d
from mps_shape_completion_msgs.msg import OccupancyStamped
import tf2_ros

rospy.init_node("test_occupancy_service")
pub = rospy.Publisher('occupancy', OccupancyStamped, queue_size=10)

services = gazebo_services.GazeboServices()
broadcaster = tf2_ros.StaticTransformBroadcaster()

res = 0.02

# this is assumed to be in frame "robot_root"
extent_3d = [0.6, 1.0, -0.2, 0.2, 0.6, 1.3]
while True:
    try:
        environment = get_environment_for_extents_3d(extent=extent_3d,
                                                     res=res,
                                                     service_provider=services,
                                                     excluded_models=['victor', 'rope_3d'])
        msg = grid_utils.environment_to_occupancy_msg(environment)
        print(msg.header.stamp)

        grid_utils.send_occupancy_tf(broadcaster, environment)
        pub.publish(msg)

        sleep(1.0)
    except rospy.ServiceException:
        pass
