#!/usr/bin/env python

from arc_utilities import ros_init
from link_bot_gazebo.gazebo_services import get_gazebo_processes


@ros_init.with_ros("resume_gazebo")
def main():
    gazebo_processes = get_gazebo_processes()
    for p in gazebo_processes:
        p.resume()


if __name__ == "__main__":
    main()
