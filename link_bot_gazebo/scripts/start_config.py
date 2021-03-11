#!/usr/bin/env python

import argparse

import rospy
from geometry_msgs.msg import Point
from peter_msgs.srv import DualGripperTrajectory, DualGripperTrajectoryRequest

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('x', type=float)
    parser.add_argument('y', type=float)
    parser.add_argument('z', type=float)

    args = parser.parse_args()

    rospy.init_node('start_config')
    srv = rospy.ServiceProxy('execute_dual_gripper_action', DualGripperTrajectory)
    rospy.wait_for_service('execute_dual_gripper_action')

    gripper1_point = Point()
    gripper1_point.x = args.x
    gripper1_point.y = args.y
    gripper1_point.z = args.z

    gripper2_point = Point()
    gripper2_point.x = args.x + 0.2
    gripper2_point.y = args.y
    gripper2_point.z = args.z

    req = DualGripperTrajectoryRequest()
    req.gripper1_points.append(gripper1_point)
    req.gripper2_points.append(gripper2_point)
    req.settling_time_seconds = 0.5
    srv(req)
