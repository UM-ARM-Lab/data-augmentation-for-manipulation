from typing import List

import pyjacobian_follower
import tensorflow as tf

from arm_robots.robot import MoveitEnabledRobot
from geometry_msgs.msg import Point
from moveit_msgs.msg import PlanningScene, RobotState


class AugOptIk:

    def __init__(self, robot: MoveitEnabledRobot, group_name='both_arms', position_only=True):
        self.position_only = position_only
        self.robot = robot
        self.group_name = group_name
        self.j = robot.jacobian_follower
        self.ik_params = pyjacobian_follower.IkParams(rng_dist=0.0, max_collision_check_attempts=1)
        if self.group_name == 'both_arms':
            self.tip_names = ['left_tool', 'right_tool']
        else:
            raise NotImplementedError()

    def solve(self, scene_msg: List[PlanningScene],
              joint_names,
              default_robot_positions,
              batch_size: int,
              left_target_position=None,
              right_target_position=None,
              left_target_pose=None,
              right_target_pose=None):
        """

        Args:
            scene_msg: [b] list of PlanningScene
            left_target_position:  [b,3], xyz
            right_target_position: [b,3], xyz
            left_target_pose:  [b,7] euler xyz, quat xyzw
            right_target_pose: [b,7] euler xyz, quat xyzw

        Returns:

        """
        robot_state_b: RobotState
        joint_positions = []
        reached = []

        def _pose_tensor_to_point(_positions):
            raise NotImplementedError()

        def _position_tensor_to_point(_positions):
            return Point(*_positions.numpy())

        for b in range(batch_size):
            scene_msg_b = scene_msg[b]

            default_robot_state_b = RobotState()
            default_robot_state_b.joint_state.position = default_robot_positions[b].numpy().tolist()
            default_robot_state_b.joint_state.name = joint_names
            scene_msg_b.robot_state.joint_state.position = default_robot_positions[b].numpy().tolist()
            scene_msg_b.robot_state.joint_state.name = joint_names
            if self.position_only:
                points_b = [_position_tensor_to_point(left_target_position[b]),
                            _position_tensor_to_point(right_target_position[b])]

                robot_state_b = self.j.compute_collision_free_point_ik(default_robot_state_b, points_b, self.group_name,
                                                                       self.tip_names,
                                                                       scene_msg_b, self.ik_params)
            else:
                poses_b = [left_target_pose[b], right_target_pose[b]]
                robot_state_b = self.j.compute_collision_free_pose_ik(default_robot_state_b, poses_b, self.group_name,
                                                                      self.tip_names, scene_msg_b, self.ik_params)

            reached.append(robot_state_b is not None)
            if robot_state_b is None:
                joint_position_b = default_robot_state_b.joint_state.position
            else:
                joint_position_b = robot_state_b.joint_state.position
            joint_positions.append(tf.convert_to_tensor(joint_position_b, dtype=tf.float32))

        joint_positions = tf.stack(joint_positions, axis=0)
        reached = tf.stack(reached, axis=0)
        return joint_positions, reached
