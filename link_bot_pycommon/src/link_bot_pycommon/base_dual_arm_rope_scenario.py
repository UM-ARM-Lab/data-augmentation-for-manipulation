import warnings
from typing import Dict, List, Optional

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    import moveit_commander
import ros_numpy
import rospy
from arc_utilities.listener import Listener
from arm_robots.get_moveit_robot import get_moveit_robot
from gazebo_ros_link_attacher.srv import Attach
from geometry_msgs.msg import PoseStamped
from link_bot_gazebo_python.gazebo_utils import get_gazebo_kinect_pose
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.floating_rope_scenario import FloatingRopeScenario
from link_bot_pycommon.ros_pycommon import get_environment_for_extents_3d, get_camera_params
from arm_gazebo_msgs.srv import ExcludeModels, ExcludeModelsRequest, ExcludeModelsResponse
from peter_msgs.srv import GetOverstretchingResponse, GetOverstretchingRequest
from rosgraph.names import ns_join
from sensor_msgs.msg import JointState, PointCloud2
from std_srvs.srv import Empty, EmptyRequest
from tf.transformations import quaternion_from_euler


class BaseDualArmRopeScenario(FloatingRopeScenario):
    ROPE_NAMESPACE = 'rope_3d'

    def __init__(self, robot_namespace):
        super().__init__()
        self.robot_namespace = robot_namespace
        self.service_provider = BaseServices()
        self.joint_state_viz_pub = rospy.Publisher(ns_join(self.robot_namespace, "joint_states_viz"), JointState,
                                                   queue_size=10)
        self.goto_home_srv = rospy.ServiceProxy("goto_home", Empty)
        self.cdcpd_listener = Listener("cdcpd/output", PointCloud2)
        self.cdcpd_reset_srv = rospy.ServiceProxy("cdcpd/reset", Empty)
        self.attach_srv = rospy.ServiceProxy("/link_attacher_node/attach", Attach)
        self.detach_srv = rospy.ServiceProxy("/link_attacher_node/detach", Attach)

        exclude_srv_name = ns_join(self.robot_namespace, "exclude_models_from_planning_scene")
        self.exclude_from_planning_scene_srv = rospy.ServiceProxy(exclude_srv_name, ExcludeModels)
        # FIXME: this blocks until the robot is available, we need lazy construction
        self.robot = get_moveit_robot(self.robot_namespace)

    def sample_action_for_data_collection(self,
                                          action_rng: np.random.RandomState,
                                          environment: Dict,
                                          state: Dict,
                                          action_params: Dict,
                                          stateless: Optional[bool] = False):
        res: GetOverstretchingResponse = self.overstretching_srv(GetOverstretchingRequest())
        if res.overstretched:
            return
        return self.sample_action(action_rng=action_rng,
                                  environment=environment,
                                  state=state,
                                  action_params=action_params,
                                  validate=True,
                                  stateless=stateless)

    def add_boxes_around_tools(self):
        # add spheres to prevent moveit from smooshing the rope and ends of grippers into obstacles
        self.moveit_scene = moveit_commander.PlanningSceneInterface(ns=self.robot_namespace)
        self.robust_add_to_scene(self.robot.left_tool_name, 'left_tool_box',
                                 self.robot.get_left_gripper_links())
        self.robust_add_to_scene(self.robot.right_tool_name, 'right_tool_box',
                                 self.robot.get_right_gripper_links())

    def robust_add_to_scene(self, link: str, new_object_name: str, touch_links: List[str]):
        box_pose = PoseStamped()
        box_pose.header.frame_id = link
        box_pose.pose.orientation.w = 1.0
        while True:
            self.moveit_scene.add_box(new_object_name, box_pose, size=(0.05, 0.05, 0.05))
            self.moveit_scene.attach_box(link, new_object_name, touch_links=touch_links)

            rospy.sleep(0.1)

            # Test if the box is in attached objects
            attached_objects = self.moveit_scene.get_attached_objects([new_object_name])
            is_attached = len(attached_objects.keys()) > 0

            # Note that attaching the box will remove it from known_objects
            is_known = new_object_name in self.moveit_scene.get_known_object_names()

            if is_attached and not is_known:
                break

    def on_before_get_state_or_execute_action(self):
        self.robot.connect()

    def on_before_data_collection(self, params: Dict):
        self.on_before_get_state_or_execute_action()
        self.add_boxes_around_tools()

        # Set the preferred tool orientations
        down = quaternion_from_euler(np.pi, 0, 0)
        self.robot.store_tool_orientations({
            self.robot.left_tool_name:  down,
            self.robot.right_tool_name: down,
        })

    def get_state(self):
        # TODO: this should be composed of function calls to get_state for arm_no_rope and get_state for rope?
        joint_state = self.robot.joint_state_listener.get()

        left_gripper_position, right_gripper_position = self.robot.get_gripper_positions()

        color_depth_cropped = self.get_rgbd()

        rope_state_vector = self.get_rope_state()

        cdcpd_vector = self.get_cdcpd_state()

        return {
            'joint_positions': np.array(joint_state.position),
            'joint_names':     np.array(joint_state.name),
            'left_gripper':    ros_numpy.numpify(left_gripper_position),
            'right_gripper':   ros_numpy.numpify(right_gripper_position),
            'rope':            np.array(cdcpd_vector, np.float32),
            'rgbd':            color_depth_cropped,
            'gt_rope':         np.array(rope_state_vector, np.float32),
        }

    def states_description(self) -> Dict:
        n_joints = self.robot.get_n_joints()
        return {
            'left_gripper':    3,
            'right_gripper':   3,
            'rope':            FloatingRopeScenario.n_links * 3,
            'joint_positions': n_joints,
            'rgbd':            self.IMAGE_H * self.IMAGE_W * 4,
        }

    def observations_description(self) -> Dict:
        return {
            'left_gripper':  3,
            'right_gripper': 3,
            'rgbd':          self.IMAGE_H * self.IMAGE_W * 4,
        }

    def plot_state_rviz(self, state: Dict, label: str, **kwargs):
        super().plot_state_rviz(state, label, **kwargs)
        if 'joint_positions' in state and 'joint_names' in state:
            joint_msg = JointState()
            joint_msg.header.stamp = rospy.Time.now()
            joint_msg.position = state['joint_positions']
            if isinstance(state['joint_names'][0], bytes):
                joint_names = [n.decode("utf-8") for n in state['joint_names']]
            elif isinstance(state['joint_names'][0], str):
                joint_names = [str(n) for n in state['joint_names']]
            else:
                raise NotImplementedError(type(state['joint_names'][0]))
            joint_msg.name = joint_names
            self.joint_state_viz_pub.publish(joint_msg)

    def dynamics_dataset_metadata(self):
        joint_state: JointState = self.robot.joint_state_listener.get()
        kinect_pose = get_gazebo_kinect_pose()
        kinect_params = get_camera_params(self.KINECT_NAME)
        return {
            'joint_names':   joint_state.name,
            'kinect_pose':   ros_numpy.numpify(kinect_pose),
            'kinect_params': kinect_params,
        }

    @staticmethod
    def simple_name():
        return "dual_arm"

    def get_excluded_models_for_env(self):
        exclude = ExcludeModelsRequest()
        res: ExcludeModelsResponse = self.exclude_from_planning_scene_srv(exclude)
        return res.all_model_names

    def initial_obstacle_poses_with_noise(self, env_rng: np.random.RandomState, obstacles: List):
        raise NotImplementedError()

    def execute_action(self, action: Dict):
        left_gripper_points = [action['left_gripper_position']]
        right_gripper_points = [action['right_gripper_position']]
        tool_names = [self.robot.left_tool_name, self.robot.right_tool_name]
        grippers = [left_gripper_points, right_gripper_points]
        traj, result = self.robot.follow_jacobian_to_position("both_arms", tool_names, grippers, vel_scaling=0.1)

        rope_settling_time = action.get('settling_time', 1.0)
        rospy.sleep(rope_settling_time)

        return traj, result

    def get_environment(self, params: Dict, **kwargs):
        res = params.get("res", 0.01)
        return get_environment_for_extents_3d(extent=params['extent'],
                                              res=res,
                                              service_provider=self.service_provider,
                                              excluded_models=self.get_excluded_models_for_env())

    @staticmethod
    def robot_name():
        raise NotImplementedError()

    def reset_cdcpd(self):
        self.cdcpd_reset_srv(EmptyRequest())
