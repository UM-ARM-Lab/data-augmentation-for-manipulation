#!/usr/bin/env python3
from time import sleep

import roscpp_initializer

import rospy
rospy.init_node("test_heartbeat")

from link_bot_pycommon.heartbeat import HeartBeat

print("running test_heartbeat")
h = HeartBeat(1)
sleep(5)  # heartbeat runs while we sleep

print("infinite C++ loop")
roscpp_initializer.infinite_loop()  # heartbeat won't be able to run during this
