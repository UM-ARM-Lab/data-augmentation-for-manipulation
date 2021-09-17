from typing import Dict

import numpy as np

import ros_numpy
from link_bot_pycommon.base_dual_arm_rope_scenario import BaseDualArmRopeScenario
from link_bot_pycommon.get_cdcpd_state import GetCdcpdState
from link_bot_pycommon.get_joint_state import GetJointState


class DualArmRealValRopeScenario(BaseDualArmRopeScenario):
    COLOR_IMAGE_TOPIC = "/kinect2_tripodA/qhd/image_color_rect"
    DEPTH_IMAGE_TOPIC = "/kinect2_tripodA/qhd/image_depth_rect"

    def __init__(self):
        super().__init__('hdt_michigan')
        self.my_closed = 0.015
        self.get_joint_state = GetJointState(self.robot)
        self.get_cdcpd_state = GetCdcpdState(self.tf)

        self.reset_move_group = 'both_arms'

    def on_before_data_collection(self, params: Dict):
        current_joint_positions = np.array(self.robot.get_joint_positions(self.robot.get_both_arm_joints()))
        near_start = np.max(np.abs(np.array(params['reset_joint_config']) - current_joint_positions)) < 0.02
        grippers_are_closed = self.robot.is_left_gripper_closed() and self.robot.is_right_gripper_closed()
        if not near_start or not grippers_are_closed:
            # let go
            self.robot.open_left_gripper()

            # move to init positions
            self.robot.plan_to_joint_config("both_arms", params['reset_joint_config'])

            self.robot.speak("press enter to close grippers")
            input("press enter to close grippers")

        self.robot.speak("press enter to begin")
        input("press enter to begin")

    def on_before_get_state_or_execute_action(self):
        self.robot.connect()
        self.add_boxes_around_tools()

    def get_state(self):
        state = {}
        state.update(self.get_robot_state.get_state())
        state.update(self.get_cdcpd_state.get_state())
        return state

    def get_excluded_models_for_env(self):
        return []

    def randomize_environment(self, env_rng: np.random.RandomState, params: Dict):
        self.robot.set_left_gripper(1.0)
        self.robot.plan_to_joint_config(self.reset_move_group, dict(params['real_val_rope_reset_joint_config']))

        while True:
            # find rope point from rope tracking
            # FIXME: do rope tracking
            # sensed_rope_state = self.get_cdcpd_state()
            # rope_point_to_grasp = sensed_rope_state[0]

            # FIXME: just debugging
            rope_point_to_pre_grasp = ros_numpy.numpify(self.robot.get_link_pose(self.robot.right_tool_name).position)
            rope_point_to_pre_grasp[2] = 0.3

            # go a little past...
            rope_point_to_pre_grasp = rope_point_to_pre_grasp + np.array([0.08, -0.08, 0.05])

            rope_point_to_grasp = rope_point_to_pre_grasp + np.array([-0.15, 0.15, 0.0])
            right_pose = self.robot.get_link_pose(self.robot.right_tool_name)
            right_position = ros_numpy.numpify(right_pose.position)

            self.robot.set_left_gripper(1.0)

            # move to pre-grasp config
            tool_names = [self.robot.left_tool_name, self.robot.right_tool_name]
            self.robot.store_current_tool_orientations(tool_names)
            self.robot.follow_jacobian_to_position(group_name='both_arms',
                                                   tool_names=tool_names,
                                                   points=[[rope_point_to_pre_grasp], [right_position]])

            self.robot.set_left_gripper(0.173)
            # move "backwards"
            self.robot.follow_jacobian_to_position(group_name='both_arms',
                                                   tool_names=tool_names,
                                                   points=[[rope_point_to_grasp], [right_position]])

            self.robot.set_left_gripper(self.my_closed)
            grasped = self.is_left_grasped()
            if grasped:
                break

    def is_left_grasped(self):
        positions = self.robot.get_joint_positions(['leftgripper', 'leftgripper2'])
        p1, p2 = positions
        return p1 > self.my_closed + 0.01 and p2 > self.my_closed + 0.01
