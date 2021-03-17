#!/usr/bin/env python
import argparse

import colorama

import rospy
from arc_utilities import ros_init
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_pycommon.get_scenario import get_scenario


@ros_init.with_ros("restore_gazebo")
def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile")

    args = parser.parse_args()

    scenario = get_scenario("dual_arm_rope_sim_val")
    scenario.on_before_get_state_or_execute_action()

    gazebo_service_provider = GazeboServices()

    params = {
        'environment_randomization': {
            'type':          'jitter',
            'nominal_poses': {
                'long_hook1': None,
            },
        },
    }

    scenario.restore_from_bag(gazebo_service_provider, params, args.bagfile)

    rospy.loginfo("done")


if __name__ == "__main__":
    main()
