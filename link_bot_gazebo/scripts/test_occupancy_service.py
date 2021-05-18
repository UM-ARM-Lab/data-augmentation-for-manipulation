#!/usr/bin/env python
from time import sleep

import numpy as np

import rospy
import tf2_ros
from arc_utilities import ros_init
from jsk_recognition_msgs.msg import BoundingBox
from link_bot_gazebo import gazebo_services
from link_bot_pycommon import grid_utils
from link_bot_pycommon.bbox_visualization import extent_to_bbox
from link_bot_pycommon.get_occupancy import get_environment_for_extents_3d
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped


@ros_init.with_ros("test_occupancy_service")
def main():
    pub = rospy.Publisher('occupancy', VoxelgridStamped, queue_size=10)
    bbox_pub = rospy.Publisher('bbox', BoundingBox, queue_size=10)

    services = gazebo_services.GazeboServices()
    broadcaster = tf2_ros.StaticTransformBroadcaster()

    res = 0.02

    # this is assumed to be in frame "robot_root"
    s = 0
    t = 0
    while True:
        dx = np.sin(t) * 2e-2
        dy = np.sin(2*t + 0.1) * 0e-2
        dz = np.sin(3*t + 0.2) * 2e-2
        t += 0.05
        extent_3d = [-0.10 + dx, 0.0 + dx, 0.50 + dy, 0.54 + dy, 0.20 + dz, 0.25 + dz]

        try:
            environment = get_environment_for_extents_3d(extent=extent_3d,
                                                         res=res,
                                                         service_provider=services,
                                                         excluded_models=['hdt_michigan', 'rope_3d'])
            msg = grid_utils.environment_to_vg_msg(environment)
            bbox_marker = extent_to_bbox(extent_3d)
            bbox_marker.header.frame_id = 'robot_root'
            bbox_pub.publish(bbox_marker)
            print(msg.header.stamp)

            grid_utils.send_voxelgrid_tf(broadcaster, environment)
            pub.publish(msg)

            sleep(0.1)
        except rospy.ServiceException:
            pass


if __name__ == '__main__':
    main()
