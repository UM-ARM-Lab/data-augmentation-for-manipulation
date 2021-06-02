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
        left_gripper = self.robot.jacobian_follower.fk(robot_state, "left_tool")
        right_gripper = self.robot.jacobian_follower.fk(robot_state, "right_tool")
        return {
            'joint_positions': np.array(joint_state.position, np.float32),
            'joint_names':     np.array(joint_state.name),
            'left_gripper':    ros_numpy.numpify(left_gripper.position).astype(np.float32),
            'right_gripper':    ros_numpy.numpify(right_gripper.position).astype(np.float32),
        }
