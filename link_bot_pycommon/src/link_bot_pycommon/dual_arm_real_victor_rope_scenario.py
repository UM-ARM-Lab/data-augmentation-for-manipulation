from typing import Dict

import numpy as np
import ros_numpy

from arm_robots.victor import Victor
from link_bot_pycommon.base_dual_arm_rope_scenario import BaseDualArmRopeScenario
from victor_hardware_interface_msgs.msg import ControlMode


class DualArmRealVictorRopeScenario(BaseDualArmRopeScenario):
    COLOR_IMAGE_TOPIC = "/kinect2_victor_head/qhd/image_color_rect"
    DEPTH_IMAGE_TOPIC = "/kinect2_victor_head/qhd/image_depth_rect"

    def __init__(self):
        super().__init__('victor')
        # FIXME: lazy construction
        self.victor = Victor()

    def on_before_data_collection(self, params: Dict):
        left_res, right_res = self.victor.base_victor.set_control_mode(ControlMode.JOINT_IMPEDANCE)
        if not left_res.success or not right_res.success:
            raise RuntimeError("Failed to switch into impedance mode")

        current_joint_positions = np.array(self.robot.get_joint_positions(self.robot.get_both_arm_joints()))
        near_start = np.max(np.abs(np.array(params['reset_joint_config']) - current_joint_positions)) < 0.02
        grippers_are_closed = self.victor.base_victor.is_left_gripper_closed() and self.victor.base_victor.is_right_gripper_closed()
        if not near_start or not grippers_are_closed:
            # let go
            self.robot.open_left_gripper()

            # move to init positions
            self.robot.plan_to_joint_config("both_arms", params['reset_joint_config'])

            self.robot.speak("press enter to close grippers")
            input("press enter to close grippers")

        self.robot.speak("press enter to begin")
        input("press enter to begin")

    def on_after_data_collection(self, params):
        self.victor.speak("Phew, I'm done. That was a lot of work! I feel like I've learned so much already.")

    def get_state(self):
        # TODO: this should be composed of function calls to get_state for arm_no_rope and get_state for rope?
        joint_state = self.robot.joint_state_listener.get()

        left_gripper_position, right_gripper_position = self.robot.get_gripper_positions()

        color_depth_cropped = self.get_rgbd()

        # rope_state_vector = self.get_rope_state()

        return {
            'joint_positions': joint_state.position,
            'joint_names': joint_state.name,
            'left_gripper': ros_numpy.numpify(left_gripper_position),
            'right_gripper': ros_numpy.numpify(right_gripper_position),
            'rgbd': color_depth_cropped,
            # 'rope': np.array(rope_state_vector, np.float32),
        }

    def states_description(self) -> Dict:
        n_joints = self.robot.get_num_joints()
        return {
            'left_gripper': 3,
            'right_gripper': 3,
            # 'rope': FloatingRopeScenario.n_links * 3,
            'joint_positions': n_joints,
            'rgbd': self.IMAGE_H * self.IMAGE_W * 4,
        }
