from typing import Dict

import numpy as np
from halo import Halo

import ros_numpy
import rosbag
import rospy
from arc_utilities.ros_helpers import get_connected_publisher
from control_msgs.msg import FollowJointTrajectoryResult as FJTR
from geometry_msgs.msg import Pose, Point, Quaternion
from link_bot_gazebo.gazebo_services import GazeboServices, gz_scope
from link_bot_pycommon.base_dual_arm_rope_scenario import BaseDualArmRopeScenario, robot_state_msg_from_state_dict
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.dual_arm_rope_action import dual_arm_rope_execute_action
from moveit_msgs.msg import DisplayRobotState
from peter_msgs.srv import *
from ros_numpy import msgify
from rosgraph.names import ns_join
from sensor_msgs.msg import JointState
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker


class SimDualArmRopeScenario(BaseDualArmRopeScenario):

    def __init__(self, robot_namespace):
        super().__init__(robot_namespace)

        self.reset_move_group = 'both_arms'
        self.service_provider = GazeboServices()

        self.set_rope_end_points_srv = rospy.ServiceProxy(ns_join(self.ROPE_NAMESPACE, "set"), Position3DAction)

    def execute_action(self, environment, state, action: Dict):
        return dual_arm_rope_execute_action(self.robot, self.tf, environment, state, action)

    def on_before_get_state_or_execute_action(self):
        super().on_before_get_state_or_execute_action()

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
        self.open_grippers_if_not_grasping()
        self.grasp_rope_endpoints()

        # randomize the object configurations
        er_type = params['environment_randomization']['type']
        if er_type == 'random':
            valid = False
            while not valid:
                random_object_poses = self.random_new_object_poses(env_rng, params)
                self.set_object_poses(random_object_poses)
                valid = not self.is_object_robot_collision(params)
        elif er_type == 'jitter':
            random_object_poses = self.jitter_object_poses(env_rng, params)
            self.set_object_poses(random_object_poses)
        else:
            raise NotImplementedError(er_type)

        # TODO: move the grippers again to more "random" starting configuration??

    def on_before_data_collection(self, params: Dict):
        super().on_before_data_collection(params)
        self.move_objects_out_of_scene(params)
        self.plan_to_reset_config(params)
        self.open_grippers_if_not_grasping()
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

    def open_grippers_if_not_grasping(self):
        left_end_grasped = self.robot.is_left_gripper_closed() and self.is_rope_point_attached('left')
        if not left_end_grasped:
            self.robot.open_left_gripper()
        right_end_grasped = self.robot.is_right_gripper_closed() and self.is_rope_point_attached('right')
        if not right_end_grasped:
            self.robot.open_right_gripper()

    def grasp_rope_endpoints(self, settling_time=5.0):
        self.service_provider.pause()
        self.make_rope_endpoints_follow_gripper()
        self.service_provider.play()
        rospy.sleep(settling_time)
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
        position = ros_numpy.msgify(Point, np.array([0, 10, 0]))
        orientation = ros_numpy.msgify(Quaternion, np.array([0, 0, 0, 1]))
        if 'nominal_poses' in params['environment_randomization']:
            objects = params['environment_randomization']['nominal_poses'].keys()
        elif 'objects' in params['environment_randomization']:
            objects = params['environment_randomization'].get('objects', None)
        else:
            raise NotImplementedError()
        out_of_scene_pose = Pose(position=position, orientation=orientation)
        out_of_scene_object_poses = {k: out_of_scene_pose for k in objects}
        self.set_object_poses(out_of_scene_object_poses)

    def restore_from_bag_rushed(self, service_provider: BaseServices, params: Dict, bagfile_name):
        """
        A less robust/accurate but way faster to restore
        Args:
            service_provider:
            params:
            bagfile_name:

        Returns:

        """
        self.service_provider.play()

        with rosbag.Bag(bagfile_name) as bag:
            joint_state: JointState = next(iter(bag.read_messages(topics=['joint_state'])))[1]

        joint_config = {}
        # NOTE: this will not work on victor because grippers don't work the same way
        for joint_name in self.robot.get_move_group_commander("whole_body").get_active_joints():
            index_of_joint_name_in_state_msg = joint_state.name.index(joint_name)
            joint_config[joint_name] = joint_state.position[index_of_joint_name_in_state_msg]
        self.robot.plan_to_joint_config("whole_body", joint_config)

        self.service_provider.pause()
        self.service_provider.restore_from_bag(bagfile_name, excluded_models=[self.robot_name()])
        self.service_provider.play()

    @Halo(text='Restoring', spinner='dots')
    def restore_from_bag(self, service_provider: BaseServices, params: Dict, bagfile_name, force=False):
        self.heartbeat()
        self.service_provider.play()

        self.move_objects_out_of_scene(params)
        self.detach_rope_from_grippers()

        with rosbag.Bag(bagfile_name) as bag:
            joint_state: JointState = next(iter(bag.read_messages(topics=['joint_state'])))[1]

        joint_config = {}
        # NOTE: this will not work on victor because grippers don't work the same way
        for joint_name in self.robot.get_joint_names("whole_body"):
            index_of_joint_name_in_state_msg = joint_state.name.index(joint_name)
            joint_config[joint_name] = joint_state.position[index_of_joint_name_in_state_msg]
        self.robot.plan_to_joint_config("whole_body", joint_config)

        self.service_provider.pause()
        self.service_provider.restore_from_bag(bagfile_name, excluded_models=[self.robot_name()])
        self.grasp_rope_endpoints(settling_time=1.0)
        self.service_provider.play()

    def publish_robot_state(self, joint_state):
        pub = rospy.Publisher('display_robot_state', DisplayRobotState, queue_size=10)
        display_robot_state_msg = DisplayRobotState()
        display_robot_state_msg.state.joint_state = joint_state
        display_robot_state_msg.state.joint_state.header.stamp = rospy.Time.now()
        display_robot_state_msg.state.is_diff = False
        pub.publish(display_robot_state_msg)

    @staticmethod
    def simple_name():
        return "sim_dual_arm_rope"

    def __repr__(self):
        return "SimDualArmRopeScenario"

    def is_object_robot_collision(self, params):
        """

        Returns: True if the robot and the environment are in collision

        """
        env = self.get_environment(params)
        state = self.get_state()
        # FIXME: this is super hacky, why does get state not include the attached collision objects? why do we pass in
        # a planning scene without the right robot state?
        start_state = robot_state_msg_from_state_dict(state)
        scene = env['scene_msg']
        start_state.attached_collision_objects = scene.robot_state.attached_collision_objects
        in_collision = self.robot.jacobian_follower.check_collision(scene=scene, start_state=start_state)
        return in_collision


class SimVictorDualArmRopeScenario(SimDualArmRopeScenario):

    def __init__(self):
        super().__init__('victor')
        self.left_preferred_tool_orientation = quaternion_from_euler(np.pi, 0, 0)
        self.right_preferred_tool_orientation = quaternion_from_euler(np.pi, 0, 0)

    @staticmethod
    def simple_name():
        return "dual_arm_rope_sim_victor"

    @staticmethod
    def robot_name():
        return 'victor'

    def __repr__(self):
        return "SimVictorDualArmRopeScenario"


class SimValDualArmRopeScenario(SimDualArmRopeScenario):

    def __init__(self):
        super().__init__('hdt_michigan')
        self.left_preferred_tool_orientation = quaternion_from_euler(-3 * np.pi / 4, -np.pi / 4, 0)
        self.right_preferred_tool_orientation = quaternion_from_euler(-3 * np.pi / 4, np.pi / 4, 0)

    def publish_preferred_tool_orientations(self):
        self.pub = get_connected_publisher("preferred_tool_orientation", Marker, queue_size=10)
        msg = Marker()
        msg.action = Marker.ADD
        msg.ns = 'left'
        msg.pose.orientation = msgify(Quaternion, self.left_preferred_tool_orientation)
        msg.type = Marker.ARROW
        msg.scale.x = 0.1
        msg.scale.y = 0.005
        msg.scale.z = 0.005
        msg.header.frame_id = 'robot_root'
        msg.header.stamp = rospy.Time.now()
        msg.color.r = 0
        msg.color.g = 1
        msg.color.b = 1
        msg.color.a = 1
        self.pub.publish(msg)
        msg.ns = 'right'
        msg.color.r = 1
        msg.color.g = 1
        msg.color.b = 0
        msg.pose.orientation = msgify(Quaternion, self.right_preferred_tool_orientation)
        self.pub.publish(msg)

    @staticmethod
    def simple_name():
        return "dual_arm_rope_sim_val"

    @staticmethod
    def robot_name():
        return 'hdt_michigan'

    def __repr__(self):
        return "SimValDualArmRopeScenario"
