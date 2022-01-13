from copy import deepcopy
from typing import Dict

import numpy as np

import ros_numpy
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion
from link_bot_pycommon.base_dual_arm_rope_scenario import BaseDualArmRopeScenario
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.dual_arm_rope_action import dual_arm_rope_execute_action
from link_bot_pycommon.get_cdcpd_state import GetCdcpdState
from link_bot_pycommon.get_joint_state import GetJointState
from moveit_msgs.msg import MotionPlanRequest, Constraints, OrientationConstraint, \
    PositionConstraint, JointConstraint, MoveItErrorCodes, PlanningScene
from moveit_msgs.srv import GetMotionPlan, GetMotionPlanResponse
from rosgraph.names import ns_join
from tf.transformations import quaternion_from_euler


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
        self.get_cdcpd_state = GetCdcpdState(self.tf)

        self.reset_move_group = 'both_arms'

        self.plan_srv = rospy.ServiceProxy("/hdt_michigan/plan_kinematic_path", GetMotionPlan)

    def execute_action(self, environment, state, action: Dict):
        action_fk = self.action_relative_to_fk(action, state)
        return dual_arm_rope_execute_action(self.robot,
                                            self.tf,
                                            environment,
                                            state,
                                            action_fk,
                                            vel_scaling=1.0,
                                            check_overstretching=False)

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
        input("press enter to begin")

    def get_state(self):
        # FIXME: seems like cdcpd is laggy, so delay here?
        rospy.sleep(5)

        state = {}
        state.update(self.get_robot_state.get_state())
        state.update(self.get_cdcpd_state.get_state())
        state['left_gripper'] = self.tf.get_transform("world", "mocap_RightHand0_RightHand0")[:3, 3]
        state['right_gripper'] = self.tf.get_transform("world", "mocap_Pelvis1_Pelvis1")[:3, 3]

        return state

    def get_excluded_models_for_env(self):
        return []

    def restore_from_bag(self, service_provider: BaseServices, params: Dict, bagfile_name):
        service_provider.restore_from_bag(bagfile_name)

        joint_names = self.robot.get_joint_names('both_arms')
        current_joint_positions = np.array(self.robot.get_joint_positions(joint_names))
        reset_joint_dict = {n: params['real_val_rope_reset_joint_config'][n] for n in joint_names}
        reset_joint_config = np.array(list(reset_joint_dict.values()))
        near_start = np.max(np.abs(reset_joint_config - current_joint_positions)) < 0.02
        grippers_are_closed = self.robot.is_left_gripper_closed() and self.robot.is_right_gripper_closed()
        if not near_start or not grippers_are_closed:
            # open gripper
            self.robot.set_left_gripper(1.0)

            # move to init positions
            self.robot.plan_to_joint_config("both_arms", reset_joint_dict)

            print("Use the gamepad to close the left gripper")
            input("press enter to begin")

    def restore_from_bag_v2(self, service_provider: BaseServices, params: Dict, bagfile_name):
        service_provider.restore_from_bag(bagfile_name)

        # reset
        root = self.robot.robot_commander.get_root_link()
        self.robot.set_left_gripper(1.0)
        self.robot.plan_to_joint_config("both_arms", params['real_val_rope_reset_joint_config'])
        tool_names = [self.robot.left_tool_name, self.robot.right_tool_name]
        left_tool_orientation = self.robot.stored_tool_orientations['left_tool']
        right_tool_orientation = self.robot.stored_tool_orientations['right_tool']
        up_and_away_position = self.tf.get_transform("world", "mocap_right_tool")[:3, 3]

        robot_state = self.robot.get_state()
        scene_msg = rospy.wait_for_message(ns_join(self.robot_namespace, 'planning_scene'), PlanningScene)

        while True:
            # FIXME: just debugging
            rope_point_to_pre_grasp = deepcopy(up_and_away_position)
            rope_point_to_pre_grasp[2] -= 0.83

            # go a little past...
            rope_point_to_pre_grasp = rope_point_to_pre_grasp + np.array([0.08, 0.08, 0.05])
            rope_point_to_pre_grasp_root = self.point_to_root(rope_point_to_pre_grasp, 'hdt_michigan_root')
            rope_point_to_pre_grasp_pose = Pose()
            rope_point_to_pre_grasp_pose.position = ros_numpy.msgify(Point, rope_point_to_pre_grasp_root)
            rope_point_to_pre_grasp_pose.orientation = ros_numpy.msgify(Quaternion, left_tool_orientation)

            up_and_away_position_root = self.point_to_root(up_and_away_position, 'hdt_michigan_root')
            up_and_away_pose = Pose()
            up_and_away_pose.position = ros_numpy.msgify(Point, up_and_away_position_root)
            up_and_away_pose.orientation = ros_numpy.msgify(Quaternion, right_tool_orientation)

            self.tf.send_transform_from_pose_msg(rope_point_to_pre_grasp_pose, root, 'left_goal')
            self.tf.send_transform_from_pose_msg(up_and_away_pose, root, 'right_goal')

            ik_sln = self.robot.jacobian_follower.compute_collision_free_pose_ik(robot_state,
                                                                                 [rope_point_to_pre_grasp_pose,
                                                                                  up_and_away_pose],
                                                                                 'both_arms',
                                                                                 tool_names,
                                                                                 scene_msg)
            self.robot.display_robot_state(ik_sln, 'ik_sln')

            right_ee_position_path_constraint = PositionConstraint()
            right_ee_position_path_constraint.header.frame_id = 'robot_root'
            right_ee_position_path_constraint.link_name = 'right_tool'
            right_ee_position_path_constraint.target_point_offset = rope_point_to_pre_grasp_pose

            # Constraints:
            move_group = self.robot.get_move_group_commander("both_arms")
            req = MotionPlanRequest()
            req.group_name = 'both_arms'
            #  - goal joint config, which we solved for given a desired pose for the right tool
            joint_goal_constraints = Constraints()
            for name in move_group.get_active_joints():
                i = ik_sln.joint_state.name.index(name)
                value = ik_sln.joint_state.position[i]
                joint_goal_constraint = JointConstraint()
                joint_goal_constraint.position = value
                joint_goal_constraint.joint_name = name
                joint_goal_constraint.weight = 1.0
                joint_goal_constraints.joint_constraints.append(joint_goal_constraint)
            req.goal_constraints.append(joint_goal_constraints)
            #  - right tool maintains orientation the whole time
            right_ee_orientation_path_constraint = OrientationConstraint()
            right_ee_orientation_path_constraint.header.frame_id = 'robot_root'
            right_ee_orientation_path_constraint.link_name = 'right_tool'
            right_ee_orientation_path_constraint.weight = 1.0
            right_ee_orientation_path_constraint.orientation = ros_numpy.msgify(Quaternion, left_tool_orientation)
            right_ee_orientation_path_constraint.absolute_x_axis_tolerance = 0.1
            right_ee_orientation_path_constraint.absolute_y_axis_tolerance = 0.1
            right_ee_orientation_path_constraint.absolute_z_axis_tolerance = 0.1
            right_ee_orientation_path_constraint.parameterization = OrientationConstraint.XYZ_EULER_ANGLES
            req.path_constraints.orientation_constraints.append(right_ee_orientation_path_constraint)
            #  - left tool maintains orientation the whole time
            # left_ee_orientation_path_constraint = OrientationConstraint()
            # req.path_constraints.orientation_constraints.append(left_ee_orientation_path_constraint)

            res: GetMotionPlanResponse = self.plan_srv(req)
            if res.motion_plan_response.error_code.val != MoveItErrorCodes.SUCCESS:
                print("Error!")
            else:
                self.robot.display_robot_traj(res.motion_plan_response.trajectory, "traj")

            # self.robot.plan_to_joint_config("both_arms", ik_sln)

            rope_point_to_grasp = rope_point_to_pre_grasp + np.array([-0.15, 0.15, 0.0])
            right_pose = self.robot.get_link_pose(self.robot.right_tool_name)
            right_position = ros_numpy.numpify(right_pose.position)

            self.robot.set_left_gripper(1.0)

            # move to pre-grasp config
            self.robot.follow_jacobian_to_position(group_name='both_arms',
                                                   tool_names=tool_names,
                                                   points=[[rope_point_to_pre_grasp], [right_position]])

            self.robot.set_left_gripper(0.15)
            # move "backwards"
            self.robot.follow_jacobian_to_position(group_name='both_arms',
                                                   tool_names=tool_names,
                                                   points=[[rope_point_to_grasp], [right_position]])

            self.robot.set_left_gripper(self.my_closed)
            grasped = self.is_left_grasped()
            if grasped:
                break

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
