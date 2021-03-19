from typing import Dict

import rospy
from moveit_msgs.msg import PlanningScene
from moveit_msgs.srv import GetPlanningSceneResponse, GetPlanningSceneRequest, GetPlanningScene
from rosgraph.names import ns_join


class MoveitPlanningSceneScenarioMixin:
    def __init__(self, robot_namespace: str):
        self.robot_namespace = robot_namespace

        self.get_planning_scene_srv = None
        self.get_planning_scene_req = GetPlanningSceneRequest()
        # all 1's in binary, see http://docs.ros.org/en/noetic/api/moveit_msgs/html/msg/PlanningSceneComponents.html
        self.get_planning_scene_req.components.components = 2 ** 10 - 1
        self.planning_scene_service_name = ns_join(self.robot_namespace, 'get_planning_scene')
        self.get_planning_scene_srv = rospy.ServiceProxy(self.planning_scene_service_name, GetPlanningScene)
        self.planning_scene_viz_pub = rospy.Publisher('planning_scene_viz', PlanningScene, queue_size=10)

    def get_environment(self):
        try:
            rospy.wait_for_service(self.planning_scene_service_name, timeout=1)
            scene_msg: GetPlanningSceneResponse = self.get_planning_scene_srv(self.get_planning_scene_req)
            return {'scene_msg': scene_msg.scene}
        except (rospy.ROSException, rospy.ROSInternalException):  # on timeout
            pass

        return {}

    def plot_environment_rviz(self, environment: Dict, **kwargs):
        if 'scene_msg' in environment:
            scene_msg: PlanningScene = environment['scene_msg']
            self.planning_scene_viz_pub.publish(scene_msg)
