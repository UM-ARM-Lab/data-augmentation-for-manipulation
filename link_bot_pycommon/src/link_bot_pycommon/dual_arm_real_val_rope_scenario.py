from copy import deepcopy
from typing import Dict

import numpy as np

import ros_numpy
import rospy
from arm_robots.robot import RobotPlanningError
from link_bot_pycommon.base_dual_arm_rope_scenario import BaseDualArmRopeScenario
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.dual_arm_rope_action import dual_arm_rope_execute_action
from link_bot_pycommon.get_cdcpd_state import GetCdcpdState
from link_bot_pycommon.get_joint_state import GetJointState
from moveit_msgs.srv import GetMotionPlan
from tf.transformations import quaternion_from_euler


def wiggle_positions(current, n, s=0.04):
    rng = np.random.RandomState(0)
    for i in range(n):
        delta = rng.uniform([-s, -s, -s * 0.5], [s, s, s])
        yield current + delta


class DualArmRealValRopeScenario(BaseDualArmRopeScenario):
    real = True

    COLOR_IMAGE_TOPIC = "/kinect2_tripodA/qhd/image_color_rect"
    DEPTH_IMAGE_TOPIC = "/kinect2_tripodA/qhd/image_depth_rect"

    def __init__(self):
        super().__init__('hdt_michigan')
        self.left_preferred_tool_orientation = quaternion_from_euler(-1.779, -1.043, -1.408)
        self.right_preferred_tool_orientation = quaternion_from_euler(np.pi, -1.408, 0)

        self.my_closed = -0.11
        self.get_joint_state = GetJointState(self.robot)
        self.root_link = self.robot.robot_commander.get_root_link()
        self.get_cdcpd_state = GetCdcpdState(self.tf, self.root_link)

        self.reset_move_group = 'both_arms'

        self.plan_srv = rospy.ServiceProxy("/hdt_michigan/plan_kinematic_path", GetMotionPlan)

    def execute_action(self, environment, state, action: Dict):
        action_fk = self.action_relative_to_fk(action, state)
        try:
            dual_arm_rope_execute_action(self.robot,
                                         self.tf,
                                         environment,
                                         state,
                                         action_fk,
                                         vel_scaling=1.0,
                                         check_overstretching=False)
        except RuntimeError as e:
            print(e)

        # NOTE sleeping because CDCPD is laggy.
        #  We sleep here instead of in get_state because we only need to sleep after we've moved
        rospy.sleep(4)

    def action_relative_to_fk(self, action, state):
        robot_state = self.get_robot_state.get_state()
        # so state gets the gripper positions via the mocap markers
        left_gripper_position_mocap = state['left_gripper']
        right_gripper_position_mocap = state['right_gripper']
        left_gripper_delta_position = action['left_gripper_position'] - left_gripper_position_mocap
        # whereas this is via fk
        left_gripper_position_fk = robot_state['left_gripper']
        right_gripper_delta_position = action['right_gripper_position'] - right_gripper_position_mocap
        right_gripper_position_fk = robot_state['right_gripper']
        action_fk = {
            'left_gripper_position':  left_gripper_position_fk + left_gripper_delta_position,
            'right_gripper_position': right_gripper_position_fk + right_gripper_delta_position,
        }
        self.tf.send_transform(action_fk['left_gripper_position'], [0, 0, 0, 1], parent=self.root_link,
                               child='left_gripper_position_fk', is_static=True)
        self.tf.send_transform(action_fk['right_gripper_position'], [0, 0, 0, 1], parent=self.root_link,
                               child='right_gripper_position_fk', is_static=True)
        return action_fk

    def on_before_data_collection(self, params: Dict):
        super().on_before_data_collection(params)

        joint_names = self.robot.get_joint_names('both_arms')
        current_joint_positions = np.array(self.robot.get_joint_positions(joint_names))
        reset_joint_config = np.array([params['reset_joint_config'][n] for n in joint_names])
        near_start = np.max(np.abs(reset_joint_config - current_joint_positions)) < 0.02
        grippers_are_closed = self.robot.is_left_gripper_closed() and self.robot.is_right_gripper_closed()
        if not near_start or not grippers_are_closed:
            # let go
            # self.robot.open_left_gripper()

            # move to init positions
            self.robot.plan_to_joint_config("both_arms", reset_joint_config.tolist())

            self.robot.speak("press enter to close grippers")
            print("Use the gamepad to close the left gripper")

        self.robot.speak("press enter to begin")
        while True:
            k = input("Ready to begin? [y]")
            if k in ['y', 'Y']:
                break
        print("Done.")

    def get_state(self):
        state = {}
        state.update(self.get_robot_state.get_state())
        state.update(self.get_cdcpd_state.get_state())
        # I'm pretty sure that specifying time as now() is necessary to ensure we get the absolute latest transform
        left_gripper_mocap = "mocap_RightHand0_RightHand0"
        right_gripper_mocap = "mocap_Pelvis1_Pelvis1"
        state['left_gripper'] = self.tf.get_transform(self.root_link, left_gripper_mocap)[:3, 3]
        state['right_gripper'] = self.tf.get_transform(self.root_link, right_gripper_mocap)[:3, 3]

        return state

    def get_excluded_models_for_env(self):
        return []

    def restore_from_bag_alt(self, service_provider: BaseServices, params: Dict, bagfile_name):
        service_provider.restore_from_bag(bagfile_name)

        joint_names = self.robot.get_joint_names('both_arms')
        current_joint_positions = np.array(self.robot.get_joint_positions(joint_names))
        reset_joint_dict = {n: params['real_val_rope_reset_joint_config'][n] for n in joint_names}
        reset_joint_config = np.array(list(reset_joint_dict.values()))
        near_start = np.max(np.abs(reset_joint_config - current_joint_positions)) < 0.02
        grippers_are_closed = self.robot.is_left_gripper_closed() and self.robot.is_right_gripper_closed()
        if not near_start or not grippers_are_closed:
            self.robot.set_left_gripper(0.1)

            # move to init positions
            self.robot.plan_to_joint_config("both_arms", reset_joint_dict)

            print("Use the gamepad to close the left gripper.")
            while True:
                k = input("Done? [y]")
                if k in ['y', 'Y']:
                    break
            print("Done.")

    def restore_from_bag(self, service_provider: BaseServices, params: Dict, bagfile_name):
        service_provider.restore_from_bag(bagfile_name)

        # reset
        reset_config = dict(params['real_val_rope_reset_joint_config'])

        current_joint_positions = np.array(self.robot.get_joint_positions(reset_config.keys()))
        reset_joint_positions = np.array(list(reset_config.values()))
        near_start = np.max(np.abs(reset_joint_positions - current_joint_positions)) < 0.02
        grippers_are_closed = self.robot.is_left_gripper_closed() and self.robot.is_right_gripper_closed()
        if near_start and grippers_are_closed:
            return

        # move to reset position
        graph_rope_config = dict(params['real_val_rope_reset_joint_config2'])
        self.robot.plan_to_joint_config("both_arms", graph_rope_config)

        rospy.sleep(5)

        # wiggle around
        tool_names = ['left_tool', 'right_tool']

        old_tool_orientations = deepcopy(self.robot.stored_tool_orientations)
        self.robot.store_current_tool_orientations(tool_names)
        current_left_pos = ros_numpy.numpify(self.robot.get_link_pose('left_tool').position)
        current_right_pos = ros_numpy.numpify(self.robot.get_link_pose('right_tool').position)

        for p in wiggle_positions(current_left_pos, 20):
            try:
                self.robot.follow_jacobian_to_position('both_arms', tool_names, [[p], [current_right_pos]])
            except RobotPlanningError:
                pass

        # move up
        left_up = ros_numpy.numpify(self.robot.get_link_pose('left_tool').position) + np.array([0, 0, 0.1])
        self.robot.follow_jacobian_to_position('both_arms', tool_names, [[left_up], [current_right_pos]])

        # go to the start config
        self.robot.plan_to_joint_config("both_arms", reset_config)

        # restore old tool orientations
        if old_tool_orientations is not None:
            self.robot.store_tool_orientations(old_tool_orientations)

        # wait for CDCPD to catch up
        rospy.sleep(15)

    def randomize_environment(self, env_rng: np.random.RandomState, params: Dict):
        raise NotImplementedError()

    def is_left_grasped(self):
        positions = self.robot.get_joint_positions(['leftgripper', 'leftgripper2'])
        p1, p2 = positions
        return p1 > self.my_closed + 0.01 and p2 > self.my_closed + 0.01

    def needs_reset(self, state: Dict, params: Dict):
        # FIXME:
        return False

    def on_after_data_collection(self, params):
        self.robot.disconnect()
