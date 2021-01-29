from typing import Dict

import ros_numpy
import rospy
from actionlib_msgs.msg import GoalStatus
from arm_robots.robot import MoveitEnabledRobot
from control_msgs.msg import FollowJointTrajectoryFeedback
from peter_msgs.srv import GetOverstretching, GetOverstretchingResponse, GetOverstretchingRequest
from rosgraph.names import ns_join


def dual_arm_rope_execute_action(robot: MoveitEnabledRobot, action: Dict):
    start_left_gripper_position, start_right_gripper_position = robot.get_gripper_positions()
    left_gripper_points = [action['left_gripper_position']]
    right_gripper_points = [action['right_gripper_position']]
    tool_names = [robot.left_tool_name, robot.right_tool_name]
    grippers = [left_gripper_points, right_gripper_points]

    def _stop_condition(feedback):
        return overstretching_stop_condition(feedback)

    result = robot.follow_jacobian_to_position(group_name="both_arms",
                                               tool_names=tool_names,
                                               points=grippers,
                                               stop_condition=_stop_condition)

    if result.execution_result.action_client_state == GoalStatus.PREEMPTED:
        rev_grippers = [[ros_numpy.numpify(start_left_gripper_position)],
                        [ros_numpy.numpify(start_right_gripper_position)]]
        robot.follow_jacobian_to_position("both_arms", tool_names, points=rev_grippers)


def overstretching_stop_condition(feedback: FollowJointTrajectoryFeedback, rope_namespace='rope_3d'):
    overstretching_srv = rospy.ServiceProxy(ns_join(rope_namespace, "rope_overstretched"), GetOverstretching)
    res: GetOverstretchingResponse = overstretching_srv(GetOverstretchingRequest())
    rospy.logdebug(f'rope overstretching magnitude={res.magnitude}, overstretched? {res.overstretched}')
    return res.overstretched
