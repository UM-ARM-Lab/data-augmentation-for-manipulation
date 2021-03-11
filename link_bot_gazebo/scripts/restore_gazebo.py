#!/usr/bin/env python
import argparse

import colorama

import rospy
from arc_utilities import ros_init
from gazebo_msgs.srv import SetLinkState
from link_bot_gazebo.gazebo_services import GazeboServices


@ros_init.with_ros("restore_gazebo")
def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile")

    args = parser.parse_args()

    srv_name = "gazebo/set_link_state"
    rospy.loginfo(f"waiting for service {srv_name}")
    set_srv = rospy.ServiceProxy(srv_name, SetLinkState)
    set_srv.wait_for_service()

    rospy.loginfo("resetting gazebo from bag file")

    gazebo_service_provider = GazeboServices()
    gazebo_service_provider.restore_from_bag(args.bagfile)
    gazebo_service_provider.play()

    rospy.loginfo("done")


if __name__ == "__main__":
    main()
