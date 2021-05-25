import numpy as np

from arm_robots.base_robot import BaseRobot
from sensor_msgs.msg import JointState


class GetJointState:

    def __init__(self, robot: BaseRobot):
        self.robot = robot

    def get_state(self):
        joint_state: JointState = self.robot.get_joint_state_listener().get()

        return {
            'joint_positions': np.array(joint_state.position, np.float32),
            'joint_names':     np.array(joint_state.name),
        }