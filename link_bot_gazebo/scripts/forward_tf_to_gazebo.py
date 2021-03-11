#!/usr/bin/env python
import argparse
import json
import pathlib
from time import sleep

import colorama

import geometry_msgs
import rospy
import tf2_ros
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import SetModelStateRequest
from link_bot_pycommon.pycommon import default_if_none
from tf import transformations


def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("info", type=pathlib.Path)

    args = parser.parse_args()

    with args.info.open("r") as info_file:
        info = json.load(info_file)

    rospy.init_node("forward_tf_to_gazebo")
    set_state_srv = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer)
    broadcaster = tf2_ros.TransformBroadcaster()

    while not rospy.is_shutdown():
        for model_name, tf_info in info.items():
            gazebo_model_frame = f"{model_name}_gazebo"
            translation = default_if_none(tf_info['translation'], [0, 0, 0])
            yaw = tf_info['yaw']

            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = tf_info['parent_frame']
            t.child_frame_id = gazebo_model_frame
            t.transform.translation.x = translation[0]
            t.transform.translation.y = translation[1]
            t.transform.translation.z = translation[2]
            if yaw is not None:
                q = transformations.quaternion_from_euler(0, 0, yaw)
            else:
                q = [0, 0, 0, 1]
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]

            broadcaster.sendTransform(t)

            try:
                transform = buffer.lookup_transform("robot_root", gazebo_model_frame, rospy.Time(), rospy.Duration(0.1))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn(f"failed to lookup transform between world and {gazebo_model_frame}")
                continue
            set_request = SetModelStateRequest()
            set_request.model_state.model_name = model_name
            set_request.model_state.pose.position = transform.transform.translation
            set_request.model_state.pose.orientation = transform.transform.rotation
            set_state_srv(set_request)

        sleep(1.0)


if __name__ == "__main__":
    main()
