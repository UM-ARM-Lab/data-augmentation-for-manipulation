#!/usr/bin/env python
from time import sleep

import rospy
import tf2_ros
from arc_utilities import ros_init
from jsk_recognition_msgs.msg import BoundingBox
from link_bot_gazebo_python import gazebo_services
from link_bot_pycommon import grid_utils
from link_bot_pycommon.bbox_visualization import extent_to_bbox
from link_bot_pycommon.get_occupancy import get_environment_for_extents_3d
from mps_shape_completion_msgs.msg import OccupancyStamped


@ros_init.with_ros("test_occupancy_service")
def main():
    pub = rospy.Publisher('occupancy', OccupancyStamped, queue_size=10)
    bbox_pub = rospy.Publisher('bbox', BoundingBox, queue_size=10)

    services = gazebo_services.GazeboServices()
    broadcaster = tf2_ros.StaticTransformBroadcaster()

    res = 0.1

    # this is assumed to be in frame "robot_root"
    extent_3d = [-1.0, 1.0, -1.0, 1.0, 0.0, 2.0]
    while True:
        try:
            environment = get_environment_for_extents_3d(extent=extent_3d,
                                                         res=res,
                                                         service_provider=services,
                                                         excluded_models=['hdt_michigan', 'rope_3d'])
            msg = grid_utils.environment_to_occupancy_msg(environment)
            bbox_marker = extent_to_bbox(extent_3d)
            bbox_marker.header.frame_id = 'robot_root'
            bbox_pub.publish(bbox_marker)
            print(msg.header.stamp)

            grid_utils.send_occupancy_tf(broadcaster, environment)
            pub.publish(msg)

            sleep(1.0)
        except rospy.ServiceException:
            pass


if __name__ == '__main__':
    main()
