#!/usr/bin/env python
import rospy
from tf import transformations as t
from arc_utilities.tf2wrapper import TF2Wrapper


def main():
    rospy.init_node("rope_endpoint_tf")
    tf = TF2Wrapper()

    right_mocap = 'mocap_Pelvis1_Pelvis1'
    left_mocap = 'mocap_RightHand0_RightHand0'
    d = 0.016
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        _rope_endpoint_tf(tf, d, left_mocap, 'mocap_left_tool')
        _rope_endpoint_tf(tf, d, right_mocap, 'mocap_right_tool')

        r.sleep()


def _rope_endpoint_tf(tf, d, mocap_gripper_frame, rope_endpoint_frame):
    zero_q = [0, 0, 0, 1]
    zero_t = [0, 0, 0]
    mocap_to_world = tf.get_transform(parent='world', child=mocap_gripper_frame)
    world_orientation = t.quaternion_inverse(t.quaternion_from_matrix(mocap_to_world))
    tf.send_transform(zero_t, world_orientation,
                      parent=mocap_gripper_frame,
                      child=mocap_gripper_frame + '_world_orientation')
    tf.send_transform([0, 0, -d], zero_q,
                      parent=mocap_gripper_frame + '_world_orientation',
                      child=rope_endpoint_frame)


if __name__ == '__main__':
    main()
