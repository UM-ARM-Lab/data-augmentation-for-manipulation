import numpy as np

import ros_numpy
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    from arm_robots.robot import MoveitEnabledRobot


class DualArmGetGripperPositions:

    def __init__(self, robot: MoveitEnabledRobot):
        self.robot = robot

    def get_state(self):
        left_gripper_position, right_gripper_position = self.robot.get_gripper_positions()

        return {
            'left_gripper':  ros_numpy.numpify(left_gripper_position).astype(np.float32),
            'right_gripper': ros_numpy.numpify(right_gripper_position).astype(np.float32),
        }
