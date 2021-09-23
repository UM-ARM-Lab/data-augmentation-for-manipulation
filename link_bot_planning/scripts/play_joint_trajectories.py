#!/usr/bin/env python
import argparse
import pathlib
from time import sleep, time

import colorama

import actionlib
import rosbag
import rospy
from arm_video_recorder.srv import TriggerVideoRecording, TriggerVideoRecordingRequest
from control_msgs.msg import FollowJointTrajectoryAction


def main():
    colorama.init(autoreset=True)

    rospy.init_node('follow_joint_trajectory_client')

    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", type=pathlib.Path)
    parser.add_argument("--no-video", action='store_true')

    args = parser.parse_args()

    record = rospy.ServiceProxy('video_recorder', TriggerVideoRecording)

    start_msg = TriggerVideoRecordingRequest()
    parent_dir = args.bagfile.parent
    filename = f"trajectory_playback-{int(time())}.avi"
    full_filename = (parent_dir / filename).absolute().as_posix()
    if not args.no_video:
        start_msg.record = True
        start_msg.filename = full_filename
        start_msg.timeout_in_sec = 600.0
        record(start_msg)

    sleep(5.0)

    client = actionlib.SimpleActionClient('both_arms_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    client.wait_for_server()

    bag = rosbag.Bag(args.bagfile)
    for _, msg, _ in bag.read_messages():
        goal = msg.goal
        goal.trajectory.header.stamp = rospy.Time.now()
        client.send_goal(goal)
        client.wait_for_result(rospy.Duration.from_sec(600.0))
        result = client.get_result()
        if client.get_state() != actionlib.GoalStatus.SUCCEEDED:
            print("Motion failed:")
            print(client.get_goal_status_text())
            print("Aborting")
            return

    bag.close()

    sleep(5.0)

    if not args.no_video:
        stop_msg = TriggerVideoRecordingRequest()
        stop_msg.record = False
        record(stop_msg)


if __name__ == '__main__':
    main()
