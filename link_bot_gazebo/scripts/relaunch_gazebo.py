#!/usr/bin/env python
import argparse
import signal
import sys
from time import sleep

import rospy
from arc_utilities import ros_init
from link_bot_gazebo.gazebo_services import GazeboServices
from roslaunch.pmon import ProcessListener
from std_msgs.msg import Empty

exit = False


def signal_handler(sig, frame):
    global exit
    exit = True


@ros_init.with_ros("relaunch_gazebo")
def main():
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('launch')
    parser.add_argument('world')
    parser.add_argument('--gui', action='store_true')

    args = parser.parse_args(rospy.myargv(sys.argv[1:]))

    restarting_pub = rospy.Publisher("gazebo_restarting", Empty, queue_size=10)

    launch_params = {
        'launch': args.launch,
        'world':  args.world,
    }

    service_provider = GazeboServices()
    listener = ProcessListener()

    gazebo_is_dead = False

    def _on_process_died(process_name: str, exit_code: int):
        nonlocal gazebo_is_dead
        gazebo_is_dead = True
        rospy.logerr(f"Process {process_name} exited with code {exit_code}")
        restarting_pub.publish(Empty())

    listener.process_died = _on_process_died

    while not exit:
        success = service_provider.launch(launch_params, gui=args.gui, world=launch_params['world'])
        if not success:
            return

        sleep(2)
        service_provider.play()

        gazebo_is_dead = False

        service_provider.gazebo_process.pm.add_process_listener(listener)

        while not gazebo_is_dead and not exit:
            sleep(1)

        service_provider.kill()
        sleep(2)


if __name__ == "__main__":
    main()
