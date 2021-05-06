from copy import deepcopy
from typing import List

from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest, MoveGroupGoal, PlanningScene, RobotState
from sensor_msgs.msg import JointState


def make_moveit_action_goal(joint_names, joint_positions):
    goal_config_constraint = Constraints()
    for name, position in zip(joint_names, joint_positions):
        joint_constraint = JointConstraint()
        joint_constraint.joint_name = name
        joint_constraint.position = position
        goal_config_constraint.joint_constraints.append(joint_constraint)

    req = MotionPlanRequest()
    req.group_name = 'both_arms'
    req.goal_constraints.append(goal_config_constraint)

    goal = MoveGroupGoal()
    goal.request = req
    return goal


def make_joint_state(position: List, name: List[str]):
    joint_state = JointState()
    joint_state.position = position
    joint_state.name = name
    joint_state.velocity = [0.0] * len(joint_state.name)
    return joint_state


def make_robot_state_from_joint_state(scene_msg: PlanningScene, joint_state: JointState):
    return RobotState(attached_collision_objects=scene_msg.robot_state.attached_collision_objects,
                      joint_state=joint_state)


def merge_joint_state_and_scene_msg(scene_msg, joint_state):
    robot_state = make_robot_state_from_joint_state(scene_msg=scene_msg, joint_state=joint_state)
    scene_msg_with_state = deepcopy(scene_msg)
    scene_msg_with_state.robot_state.joint_state = joint_state
    return scene_msg_with_state, robot_state


def make_robot_state(scene_msg: PlanningScene, position: List, name: List[str]):
    joint_state = make_joint_state(position, name)
    robot_state = make_robot_state_from_joint_state(scene_msg, joint_state)
    return robot_state
