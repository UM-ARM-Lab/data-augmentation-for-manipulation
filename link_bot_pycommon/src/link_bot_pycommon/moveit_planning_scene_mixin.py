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
        scene_msg: GetPlanningSceneResponse = self.get_planning_scene_srv(self.get_planning_scene_req)
        scene = scene_msg.scene

        found_ee = False
        for aco in scene.robot_state.attached_collision_objects:
            for touch_link in aco.touch_links:
                if 'end_effector' in touch_link:
                    found_ee = True

        assert found_ee, "Did not find the end effectors in the touch links of the attached objects!"

        # Clearing this makes it less likely that we accidentally use the robot state in the future
        # the robot state here is not part of the environment
        scene.robot_state.joint_state.name = []
        scene.robot_state.joint_state.position = []

        return {'scene_msg': scene}

    def plot_environment_rviz(self, environment: Dict, **kwargs):
        if 'scene_msg' in environment:
            scene_msg: PlanningScene = environment['scene_msg']
            self.planning_scene_viz_pub.publish(scene_msg)
