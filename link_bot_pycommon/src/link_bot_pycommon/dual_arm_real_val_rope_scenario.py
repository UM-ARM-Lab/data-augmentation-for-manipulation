from typing import Dict

import numpy as np

import ros_numpy
from arm_robots.hdt_michigan import Val
from arm_robots.robot import RobotPlanningError
from arm_robots.robot_utils import PlanningResult, PlanningAndExecutionResult
from geometry_msgs.msg import Quaternion, Pose, Point
from link_bot_pycommon.base_dual_arm_rope_scenario import BaseDualArmRopeScenario
from link_bot_pycommon.get_cdcpd_state import GetCdcpdState
from link_bot_pycommon.get_joint_state import GetJointState
from moveit_msgs.msg import Constraints, OrientationConstraint, PositionConstraint
from tf.transformations import quaternion_from_euler


class DualArmRealValRopeScenario(BaseDualArmRopeScenario):
    COLOR_IMAGE_TOPIC = "/kinect2_tripodA/qhd/image_color_rect"
    DEPTH_IMAGE_TOPIC = "/kinect2_tripodA/qhd/image_depth_rect"

    def __init__(self):
        super().__init__('hdt_michigan')
        self.val = Val()
        self.get_joint_state = GetJointState(self.robot)
        self.get_cdcpd_state = GetCdcpdState(self.tf)

        self.reset_move_group = 'both_arms'

        self.robot.gripper_closed_position = 0.01

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
        robot: Val = self.robot
        robot.open_left_gripper()
        robot.plan_to_joint_config(self.reset_move_group, dict(params['real_val_rope_reset_joint_config']))

        while True:
            # find rope point from rope tracking
            # FIXME: do rope tracking
            # sensed_rope_state = self.get_cdcpd_state()
            # rope_point_to_grasp = sensed_rope_state[0]
            rope_point_to_grasp = np.array([0.77, -0.18, 0.3])
            right_pose = robot.get_link_pose(robot.right_tool_name)
            # convert to goal pose
            orientation = ros_numpy.msgify(Quaternion, quaternion_from_euler(0, np.pi / 2, 0))
            left_pose = Pose(position=ros_numpy.msgify(Point, rope_point_to_grasp), orientation=orientation)
            # plan to target pose. maintain the pose of the right gripper the entire time, while moving all the joints
            # and achieving the target goal pose for the left gripper
            # how to do this???

            # close gripper
            robot.close_left_gripper()
            # confirm rope is grasped, break if it is, otherwise repeat
            grasped = self.is_left_grasped()
            if grasped:
                break

        self.robot.close_left_gripper()
        pass

    def is_left_grasped(self):
        robot: Val = self.robot
        positions = robot.get_joint_positions(['leftgripper', 'leftgripper2'])
        p1, p2 = positions
        return p1 < 0.05 and p2 < 0.05

    def plan_to_pose_with_left_while_maintaining_right(self, left_pose, right_pose):
        robot: Val = self.robot
        move_group = robot.get_move_group_commander('both_arms')

        move_group.set_pose_target(left_pose, robot.left_tool_name)
        position_constraint = PositionConstraint(link_name=robot.right_tool_name, target_point_offset=right_pose.position)
        orientation_constraint = OrientationConstraint(link_name=robot.right_tool_name, orientation=right_pose.orientation)
        position_constraints = [position_constraint]
        orientation_constraints = [orientation_constraint]
        path_constraints = Constraints(position_constraints=position_constraints,
                                       orientation_constraints=orientation_constraints)
        move_group.set_path_constraints(path_constraints)

        # TODO: debug why this fails

        planning_result = PlanningResult(move_group.plan())
        if robot.raise_on_failure and not planning_result.success:
            raise RobotPlanningError(f"Plan to position failed {planning_result.planning_error_code}")

        execution_result = robot.follow_arms_joint_trajectory(planning_result.plan.joint_trajectory, None)
        return PlanningAndExecutionResult(planning_result, execution_result)
