from typing import Dict

import numpy as np

from arm_robots.hdt_michigan import Val
from link_bot_pycommon.base_dual_arm_rope_scenario import BaseDualArmRopeScenario
from link_bot_pycommon.get_cdcpd_state import GetCdcpdState
from link_bot_pycommon.get_joint_state import GetJointState


class DualArmRealValRopeScenario(BaseDualArmRopeScenario):
    COLOR_IMAGE_TOPIC = "/kinect2_tripodA/qhd/image_color_rect"
    DEPTH_IMAGE_TOPIC = "/kinect2_tripodA/qhd/image_depth_rect"

    def __init__(self):
        super().__init__('val')
        self.val = Val()
        self.get_joint_state = GetJointState(self.robot)
        self.get_cdcpd_state = GetCdcpdState(self.tf)

    def on_before_data_collection(self, params: Dict):
        current_joint_positions = np.array(self.robot.get_joint_positions(self.robot.get_both_arm_joints()))
        near_start = np.max(np.abs(np.array(params['reset_joint_config']) - current_joint_positions)) < 0.02
        grippers_are_closed = self.val.is_left_gripper_closed() and self.val.is_right_gripper_closed()
        if not near_start or not grippers_are_closed:
            # let go
            self.robot.open_left_gripper()

            # move to init positions
            self.robot.plan_to_joint_config("both_arms", params['reset_joint_config'])

            self.robot.speak("press enter to close grippers")
            input("press enter to close grippers")

        self.robot.speak("press enter to begin")
        input("press enter to begin")

    def get_state(self):
        state = {}
        state.update(self.get_joint_state.get_state())
        state.update(self.get_cdcpd_state.get_state())
        state.update(self.get_gripper_positions.get_state())
        return state
