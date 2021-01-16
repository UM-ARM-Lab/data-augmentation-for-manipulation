from typing import Dict

import numpy as np

import rosbag
import rospy
from arm_gazebo_msgs.srv import ExcludeModelsRequest
from link_bot_gazebo_python.gazebo_services import GazeboServices, gz_scope
from link_bot_pycommon.base_dual_arm_rope_scenario import BaseDualArmRopeScenario
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.dual_arm_rope_action import dual_arm_rope_execute_action
from peter_msgs.srv import *
from rosgraph.names import ns_join
from sensor_msgs.msg import JointState
from tf.transformations import quaternion_from_euler


class SimDualArmRopeScenario(BaseDualArmRopeScenario):

    def __init__(self, robot_namespace):
        super().__init__(robot_namespace)

        self.service_provider = GazeboServices()

        # register a new callback to stop when the rope is overstretched
        self.set_rope_end_points_srv = rospy.ServiceProxy(ns_join(self.ROPE_NAMESPACE, "set"), Position3DAction)

    def execute_action(self, action: Dict):
        dual_arm_rope_execute_action(self.robot, action)

    def on_before_data_collection(self, params: Dict):
        super().on_before_data_collection(params)

        # move to init positions
        self.robot.plan_to_joint_config("both_arms", params['reset_joint_config'])

        # Grasp the rope again
        self.grasp_rope_endpoints()

    def on_before_get_state_or_execute_action(self):
        self.robot.connect()

        self.robot.store_tool_orientations({
            self.robot.right_tool_name: quaternion_from_euler(np.pi, 0, 0),
            self.robot.left_tool_name : quaternion_from_euler(np.pi, 0, 0),
        })

        # Mark the rope as a not-obstacle
        exclude = ExcludeModelsRequest()
        exclude.model_names.append("rope_3d")
        self.exclude_from_planning_scene_srv(exclude)

        # register kinematic controllers for fake-grasping
        self.register_fake_grasping()

    def randomize_environment(self, env_rng: np.random.RandomState, params: Dict):
        # teleport movable objects out of the way
        self.move_objects_out_of_scene(params)

        # release the rope
        self.robot.open_left_gripper()
        self.detach_rope_from_gripper('left_gripper')

        # plan to reset joint config, we assume this will always work
        self.robot.plan_to_joint_config("both_arms", params['reset_joint_config'])

        # Grasp the rope again
        self.grasp_rope_endpoints()

        # randomize the object configurations
        random_object_poses = self.random_new_object_poses(env_rng, params)
        self.set_object_poses(random_object_poses)

        # TODO: move the grippers again to more "random" starting configuration??

    def grasp_rope_endpoints(self):
        self.robot.open_left_gripper()
        self.robot.open_right_gripper()

        self.service_provider.pause()
        self.make_rope_endpoints_follow_gripper()
        self.service_provider.play()
        rospy.sleep(5)
        self.robot.close_left_gripper()
        self.robot.close_right_gripper()

        self.reset_cdcpd()

    def move_rope_out_of_the_scene(self):
        set_req = Position3DActionRequest()
        set_req.scoped_link_name = gz_scope(self.ROPE_NAMESPACE, "left_gripper")
        set_req.position.x = 1.3
        set_req.position.y = 0.3
        set_req.position.z = 1.3
        self.pos3d.set(set_req)

        set_req = Position3DActionRequest()
        set_req.scoped_link_name = gz_scope(self.ROPE_NAMESPACE, "right_gripper")
        set_req.position.x = 1.3
        set_req.position.y = -0.3
        set_req.position.z = 1.3
        self.pos3d.set(set_req)

    def detach_rope_from_gripper(self, rope_link_name: str):
        enable_req = Position3DEnableRequest()
        enable_req.scoped_link_name = gz_scope(self.ROPE_NAMESPACE, rope_link_name)
        enable_req.enable = False
        self.pos3d.enable(enable_req)

    def detach_rope_from_grippers(self):
        self.detach_rope_from_gripper('left_gripper')
        self.detach_rope_from_gripper('right_gripper')

    def move_objects_out_of_scene(self, params: Dict):
        position = [0, 2, 0]
        orientation = [0, 0, 0, 1]
        out_of_scene_object_poses = {k: (position, orientation) for k in params['objects']}
        self.set_object_poses(out_of_scene_object_poses)

    def restore_from_bag(self, service_provider: BaseServices, params: Dict, bagfile_name):
        self.service_provider.play()

        self.move_objects_out_of_scene(params)
        self.robot.open_left_gripper()
        self.detach_rope_from_grippers()

        with rosbag.Bag(bagfile_name) as bag:
            joint_state: JointState = next(iter(bag.read_messages(topics=['joint_state'])))[1]

        joint_config = []
        for joint_name in self.robot.get_both_arm_joints():
            index_of_joint_name_in_state_msg = joint_state.name.index(joint_name)
            joint_config.append(joint_state.position[index_of_joint_name_in_state_msg])
        self.robot.plan_to_joint_config("both_arms", joint_config)

        self.service_provider.pause()
        self.service_provider.restore_from_bag(bagfile_name, excluded_models=['victor'])
        self.grasp_rope_endpoints()
        self.service_provider.play()

    @staticmethod
    def simple_name():
        return "dual_arm_rope_sim_victor"


class SimVictorDualArmRopeScenario(BaseDualArmRopeScenario):

    def __init__(self):
        super().__init__('victor')


class SimValDualArmRopeScenario(BaseDualArmRopeScenario):

    def __init__(self):
        super().__init__('hdt_michigan')
