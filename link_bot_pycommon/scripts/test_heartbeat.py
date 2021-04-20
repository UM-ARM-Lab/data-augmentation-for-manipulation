#!/usr/bin/env python3
from time import sleep

from arc_utilities import ros_init
from link_bot_pycommon.heartbeat import HeartBeat


@ros_init.with_ros("test_heartbeat")
def main():
    print("running test_heartbeat")
    h = HeartBeat(1)
    sleep(5)  # heartbeat runs while we sleep
    print("done!")

    # print("infinite C++ loop")
    # roscpp_initializer.infinite_loop()  # heartbeat won't be able to run during this


if __name__ == '__main__':
    main()
