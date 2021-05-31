import numpy as np
import transformations

import ros_numpy
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import Pose, Point, Quaternion


def transform_link_states(m: np.ndarray, link_states: LinkStates):
    link_states_aug = LinkStates()
    for name, pose, twist in zip(link_states.name, link_states.pose, link_states.twist):
        translate = ros_numpy.numpify(pose.position)
        angles = transformations.euler_from_quaternion(ros_numpy.numpify(pose.orientation))
        link_transform = transformations.compose_matrix(angles=angles, translate=translate)
        link_transform_aug = m @ link_transform
        pose_aug = Pose()
        pose_aug.position = ros_numpy.msgify(Point, transformations.translation_from_matrix(link_transform_aug))
        pose_aug.orientation = ros_numpy.msgify(Quaternion, transformations.quaternion_from_matrix(link_transform_aug))
        twist_aug = twist  # ignoring this for now, it should be near zero already
        link_states_aug.name.append(name)
        link_states_aug.pose.append(pose_aug)
        link_states_aug.twist.append(twist_aug)
    return link_states_aug