#!/usr/bin/env python
import argparse
from time import sleep

import numpy as np
import rospy

from peter_msgs.srv import ExecuteAction, ExecuteActionRequest
from link_bot_pycommon.args import my_formatter

if __name__ == '__main__':
    rospy.init_node('square')

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('--size', type=float, default=1.0)
    parser.add_argument('--time', type=float, default=1.0)

    args = parser.parse_args()

    execute_action = rospy.ServiceProxy("/execute_action", ExecuteAction)

    action = ExecuteActionRequest()

    dt = 8.0
    action.action.action = [5, 5]
    action.action.max_time_per_step = dt
    execute_action(action)

    dt = 2.0
    action.action.action = [0, 0]
    action.action.max_time_per_step = dt
    execute_action(action)
