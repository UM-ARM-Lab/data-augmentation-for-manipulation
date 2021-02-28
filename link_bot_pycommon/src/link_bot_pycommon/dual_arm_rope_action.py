from typing import Dict

import ros_numpy
import rospy
from actionlib_msgs.msg import GoalStatus
from arm_robots.robot import MoveitEnabledRobot
from peter_msgs.srv import GetOverstretching, GetOverstretchingResponse, GetOverstretchingRequest
from rosgraph.names import ns_join


def dual_arm_rope_execute_action(robot: MoveitEnabledRobot, action: Dict):
    start_left_gripper_pos, start_right_gripper_pos = robot.get_gripper_positions()
    left_gripper_points = [action['left_gripper_position']]
    right_gripper_points = [action['right_gripper_position']]
    tool_names = [robot.left_tool_name, robot.right_tool_name]
    grippers = [left_gripper_points, right_gripper_points]

    overstretching_srv = rospy.ServiceProxy(ns_join("rope_3d", "rope_overstretched"), GetOverstretching)
    res: GetOverstretchingResponse = overstretching_srv(GetOverstretchingRequest())

    if res.magnitude > 1.20:
        # just do nothing...
        rospy.logwarn("The rope is extremely overstretched -- refusing to execute action")
        return (end_trial := True)

    def _stop_condition(_):
        return overstretching_stop_condition()

    result = robot.follow_jacobian_to_position(group_name="both_arms",
                                               tool_names=tool_names,
                                               points=grippers,
                                               stop_condition=_stop_condition)

    rospy.sleep(1.0)
    res: GetOverstretchingResponse = overstretching_srv(GetOverstretchingRequest())
    if result.execution_result.action_client_state == GoalStatus.PREEMPTED or res.overstretched:
        rev_grippers = [[ros_numpy.numpify(start_left_gripper_pos)],
                        [ros_numpy.numpify(start_right_gripper_pos)]]
        robot.follow_jacobian_to_position("both_arms", tool_names, points=rev_grippers)

    return (end_trial := False)

def overstretching_stop_condition():
    overstretching_srv = rospy.ServiceProxy(ns_join("rope_3d", "rope_overstretched"), GetOverstretching)
    res: GetOverstretchingResponse = overstretching_srv(GetOverstretchingRequest())
    return res.overstretched
