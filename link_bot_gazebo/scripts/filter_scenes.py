#!/usr/bin/env python
import argparse
import pathlib

import rosbag
import rospy
from arc_utilities.listener import Listener
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.srv import SetLinkState
from link_bot_gazebo.gazebo_services import GazeboServices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=pathlib.Path)

    args = parser.parse_args()

    rospy.init_node('filter_scenes')

    link_states_listener = Listener("/gazebo/link_states", LinkStates)

    bagfile_names = list(args.dir.iterdir())
    bagfile_names = sorted(bagfile_names)
    for bagfile_name in bagfile_names:
        if bagfile_name.suffix == '.bag':
            index = int(bagfile_name.stem[-4:])
            srv_name = "gazebo/set_link_state"
            rospy.loginfo(f"waiting for service {srv_name}")
            set_srv = rospy.ServiceProxy(srv_name, SetLinkState)
            set_srv.wait_for_service()

            gazebo_service_provider = GazeboServices()
            gazebo_service_provider.restore_from_bag(bagfile_name)

            gazebo_service_provider.play()

            r = input(f"{index:4d}: fix? [N/y]")
            if r == 'Y' or r == 'y':
                # let the use fix it
                input("press enter when you're done fixing")
                links_states = link_states_listener.get()
                bagfile_name = args.dir / f'scene_{index:04d}.bag'
                rospy.loginfo(f"Saving scene to {bagfile_name}")
                with rosbag.Bag(bagfile_name, 'w') as bag:
                    bag.write('links_states', links_states)


if __name__ == '__main__':
    main()
