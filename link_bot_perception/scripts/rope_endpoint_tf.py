#!/usr/bin/env python
import argparse

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
# NOTE: this is the right one to import, don't change it
from tf import transformations as t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-left', action='store_true')
    parser.add_argument('--no-right', action='store_true')

    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("rope_endpoint_tf")
    tfw = TF2Wrapper()

    right_mocap = 'mocap_Pelvis1_Pelvis1'
    left_mocap = 'mocap_RightHand0_RightHand0'

    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        if not args.no_left:
            _rope_endpoint_tf(tfw, [-0.01, 0.01, -0.01], left_mocap, 'mocap_left_tool')
        if not args.no_right:
            _rope_endpoint_tf(tfw, [0, 0, -0.01], right_mocap, 'mocap_right_tool')

        r.sleep()


def _rope_endpoint_tf(tfw, offset, mocap_gripper_frame, rope_endpoint_frame):
    zero_q = [0, 0, 0, 1]
    zero_t = [0, 0, 0]
    mocap_to_world = tfw.get_transform('world', mocap_gripper_frame)
    world_orientation = t.quaternion_inverse(t.quaternion_from_matrix(mocap_to_world))
    tfw.send_transform(zero_t, world_orientation,
                       parent=mocap_gripper_frame,
                       child=mocap_gripper_frame + '_world_orientation')
    tfw.send_transform(offset, zero_q,
                       parent=mocap_gripper_frame + '_world_orientation',
                       child=rope_endpoint_frame)


if __name__ == '__main__':
    main()
