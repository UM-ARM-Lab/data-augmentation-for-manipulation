from typing import Dict

import numpy as np

import ros_numpy
import rosbag
import rospy
from arm_gazebo_msgs.srv import ExcludeModelsRequest
from control_msgs.msg import FollowJointTrajectoryResult as FJTR
from geometry_msgs.msg import Pose, Point, Quaternion
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

        self.reset_move_group = 'both_arms'
        self.service_provider = GazeboServices()

        self.set_rope_end_points_srv = rospy.ServiceProxy(ns_join(self.ROPE_NAMESPACE, "set"), Position3DAction)

    def execute_action(self, action: Dict):
        dual_arm_rope_execute_action(self.robot, action)

    def on_before_get_state_or_execute_action(self):
        self.robot.connect()

        self.robot.store_tool_orientations({
            self.robot.right_tool_name: quaternion_from_euler(np.pi, 0, 0),
            self.robot.left_tool_name:  quaternion_from_euler(np.pi, 0, 0),
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

        self.plan_to_reset_config(params)

        # Grasp the rope again
        self.grasp_rope_endpoints()

        # randomize the object configurations
        er_type = params['environment_randomization']['type']
        if er_type == 'random':
            random_object_poses = self.random_new_object_poses(env_rng, params)
            self.set_object_poses(random_object_poses)
        elif er_type == 'jitter':
            random_object_poses = self.jitter_object_poses(env_rng, params)
            self.set_object_poses(random_object_poses)
        else:
            raise NotImplementedError(er_type)

        # TODO: move the grippers again to more "random" starting configuration??

    def on_before_data_collection(self, params: Dict):
        super().on_before_data_collection(params)
        self.plan_to_reset_config(params)
        self.grasp_rope_endpoints()

    def plan_to_reset_config(self, params: Dict):
        result = self.robot.plan_to_joint_config(self.reset_move_group, dict(params['reset_joint_config']))
        if result.execution_result.execution_result.error_code not in [FJTR.SUCCESSFUL,
                                                                       FJTR.GOAL_TOLERANCE_VIOLATED] \
                or not result.planning_result.success:
            rospy.logfatal("Could not plan to reset joint config! Aborting")
            # by exiting here, we prevent saving bogus data
            import sys
            sys.exit(-3)
        if result.execution_result.execution_result.error_code == FJTR.GOAL_TOLERANCE_VIOLATED:
            rospy.logwarn("Goal tolerance violated while resetting?")

    def grasp_rope_endpoints(self):
        left_end_grasped = self.robot.is_left_gripper_closed() and self.is_rope_point_attached('left')
        if not left_end_grasped:
            self.robot.open_left_gripper()
        right_end_grasped = self.robot.is_right_gripper_closed() and self.is_rope_point_attached('right')
        if not right_end_grasped:
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
        position = ros_numpy.msgify(Point, np.array([0, 2, 0]))
        orientation = ros_numpy.msgify(Quaternion, np.array([0, 0, 0, 1]))
        er_params = params['environment_randomization']
        if er_params['type'] == 'random':
            objects = params['environment_randomization']['objects']
        elif er_params['type'] == 'jitter':
            objects = params['environment_randomization']['nominal_poses'].keys()
        else:
            raise NotImplementedError(er_params['type'])
        out_of_scene_pose = Pose(position=position, orientation=orientation)
        out_of_scene_object_poses = {k: out_of_scene_pose for k in objects}
        self.set_object_poses(out_of_scene_object_poses)

    def restore_from_bag(self, service_provider: BaseServices, params: Dict, bagfile_name):
        self.service_provider.play()

        self.move_objects_out_of_scene(params)
        self.robot.open_left_gripper()
        self.detach_rope_from_grippers()

        with rosbag.Bag(bagfile_name) as bag:
            joint_state: JointState = next(iter(bag.read_messages(topics=['joint_state'])))[1]

        joint_config = {}
        for joint_name in self.robot.get_both_arm_joints():
            index_of_joint_name_in_state_msg = joint_state.name.index(joint_name)
            joint_config[joint_name] = joint_state.position[index_of_joint_name_in_state_msg]
        self.robot.plan_to_joint_config("both_arms", joint_config)

        self.service_provider.pause()
        self.service_provider.restore_from_bag(bagfile_name, excluded_models=['victor'])
        self.grasp_rope_endpoints()
        self.service_provider.play()

    @staticmethod
    def simple_name():
        return "sim_dual_arm_rope"

    def __repr__(self):
        return "SimDualArmRopeScenario"


class SimVictorDualArmRopeScenario(SimDualArmRopeScenario):

    def __init__(self):
        super().__init__('victor')

    @staticmethod
    def simple_name():
        return "sim_victor_dual_arm_rope"

    def __repr__(self):
        return "SimVictorDualArmRopeScenario"


class SimValDualArmRopeScenario(SimDualArmRopeScenario):

    def __init__(self):
        super().__init__('hdt_michigan')

    @staticmethod
    def simple_name():
        return "sim_val_dual_arm_rope"

    def __repr__(self):
        return "SimValDualArmRopeScenario"
