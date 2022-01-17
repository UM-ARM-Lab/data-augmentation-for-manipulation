import warnings

import numpy as np

import ros_numpy
import rospy
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    from arm_robots.robot import MoveitEnabledRobot


class GetDualArmRobotState:

    def __init__(self, robot: MoveitEnabledRobot):
        self.robot = robot

    def get_state(self):
        joint_state: JointState = rospy.wait_for_message(self.robot.joint_states_topic, JointState)
        robot_state = RobotState(joint_state=joint_state)
        left_gripper_root = self._gripper(robot_state, 'left')
        right_gripper_root = self._gripper(robot_state, 'right')

        return {
            'joint_positions': np.array(joint_state.position, np.float32),
            'joint_names':     np.array(joint_state.name),
            'left_gripper':    left_gripper_root,
            'right_gripper':   right_gripper_root,
        }

    def _gripper(self, robot_state, name):
        while True:
            gripper = self.robot.jacobian_follower.fk(robot_state, f"{name}_tool")
            gripper_root = self.robot.tf_wrapper.transform_to_frame(gripper, self.robot.robot_commander.get_root_link())
            gripper_root = ros_numpy.numpify(gripper_root.pose.position).astype(np.float32)
            if not np.any(np.abs(gripper_root) > 1e6):
                return gripper_root
