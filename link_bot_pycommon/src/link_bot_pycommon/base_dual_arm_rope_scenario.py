import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

import rosnode
from moveit_msgs.msg import RobotState, RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    import moveit_commander
import ros_numpy
import rospy
from arc_utilities.listener import Listener
from arm_robots.get_robot import get_moveit_robot
from geometry_msgs.msg import PoseStamped
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.floating_rope_scenario import FloatingRopeScenario
from link_bot_pycommon.get_occupancy import get_environment_for_extents_3d
from arm_gazebo_msgs.srv import ExcludeModels, ExcludeModelsRequest, ExcludeModelsResponse
from rosgraph.names import ns_join
from sensor_msgs.msg import JointState, PointCloud2
from tf.transformations import quaternion_from_euler


class BaseDualArmRopeScenario(FloatingRopeScenario):
    DISABLE_CDCPD = True
    ROPE_NAMESPACE = 'rope_3d'

    def __init__(self, robot_namespace):
        super().__init__()
        self.robot_namespace = robot_namespace
        self.service_provider = BaseServices()
        self.joint_state_viz_pub = rospy.Publisher(ns_join(self.robot_namespace, "joint_states_viz"), JointState,
                                                   queue_size=10)
        self.cdcpd_listener = Listener("cdcpd/output", PointCloud2)

        # NOTE: you may want to override this for your specific robot/scenario
        self.left_preferred_tool_orientation = quaternion_from_euler(np.pi, 0, 0)
        self.right_preferred_tool_orientation = quaternion_from_euler(np.pi, 0, 0)

        self.size_of_box_around_tool_for_planning = 0.05
        exclude_srv_name = ns_join(self.robot_namespace, "exclude_models_from_planning_scene")
        self.exclude_from_planning_scene_srv = rospy.ServiceProxy(exclude_srv_name, ExcludeModels)
        # FIXME: this blocks until the robot is available, we need lazy construction
        self.robot = get_moveit_robot(self.robot_namespace, raise_on_failure=True)
        self.stored_joint_names = None

    def add_boxes_around_tools(self):
        # add spheres to prevent moveit from smooshing the rope and ends of grippers into obstacles
        self.moveit_scene = moveit_commander.PlanningSceneInterface(ns=self.robot_namespace)
        self.robust_add_to_scene(self.robot.left_tool_name, 'left_tool_box', self.robot.get_left_gripper_links())
        self.robust_add_to_scene(self.robot.right_tool_name, 'right_tool_box', self.robot.get_right_gripper_links())

    def robust_add_to_scene(self, link: str, new_object_name: str, touch_links: List[str]):
        box_pose = PoseStamped()
        box_pose.header.frame_id = link
        box_pose.pose.orientation.w = 1.0
        box_size = self.size_of_box_around_tool_for_planning
        while True:
            self.moveit_scene.add_box(new_object_name, box_pose, size=(box_size, box_size, box_size))
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

        # Mark the rope as a not-obstacle
        exclude = ExcludeModelsRequest()
        exclude.model_names.append("rope_3d")
        exclude.model_names.append(self.robot_namespace)
        self.exclude_from_planning_scene_srv(exclude)

    def on_before_data_collection(self, params: Dict):
        self.on_before_get_state_or_execute_action()
        self.add_boxes_around_tools()

        # Set the preferred tool orientations
        self.robot.store_tool_orientations({
            self.robot.left_tool_name:  self.left_preferred_tool_orientation,
            self.robot.right_tool_name: self.right_preferred_tool_orientation,
        })

    def get_n_joints(self):
        return len(self.get_joint_names())

    def get_joint_names(self):
        # by storing joint names here, we can query in the future while time is paused
        if self.stored_joint_names is None:
            self.stored_joint_names = self.robot.get_joint_names()

        return self.stored_joint_names

    def get_state(self):
        # TODO: this should be composed of function calls to get_state for arm_no_rope and get_state for rope?
        joint_state: JointState = self.robot._joint_state_listener.get()

        # FIXME: "Joint values for monitored state are requested but the full state is not known"
        for _ in range(5):
            left_gripper_position, right_gripper_position = self.robot.get_gripper_positions()
            rospy.sleep(0.02)

        # rgbd = self.get_rgbd()

        gt_rope_state_vector = self.get_rope_state()
        gt_rope_state_vector = np.array(gt_rope_state_vector, np.float32)

        if self.DISABLE_CDCPD:
            cdcpd_rope_state_vector = gt_rope_state_vector
        else:
            cdcpd_rope_state_vector = self.get_cdcpd_state()

        return {
            'joint_positions': np.array(joint_state.position),
            'joint_names':     np.array(joint_state.name),
            'left_gripper':    ros_numpy.numpify(left_gripper_position),
            'right_gripper':   ros_numpy.numpify(right_gripper_position),
            # 'rgbd':            rgbd,
            'gt_rope':         gt_rope_state_vector,
            'rope':            cdcpd_rope_state_vector,
        }

    def states_description(self) -> Dict:
        n_joints = self.robot.get_num_joints()
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

    def plot_state_rviz(self, state: Dict, **kwargs):
        super().plot_state_rviz(state, **kwargs)
        label = kwargs.pop("label", "")
        if 'joint_positions' in state and 'joint_names' in state:
            joint_state = self.joint_state_msg_from_state_dict(state)
            self.robot.display_robot_state(joint_state, label, kwargs.get("color", None))
        elif 'joint_positions' not in state:
            rospy.logwarn_throttle(10, 'no joint positions in state', logger_name=Path(__file__).stem)
        elif 'joint_names' not in state:
            rospy.logwarn_throttle(10, 'no joint names in state', logger_name=Path(__file__).stem)

    def joint_state_msg_from_state_dict(self, state):
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.position = state['joint_positions']
        if isinstance(state['joint_names'][0], bytes):
            joint_names = [n.decode("utf-8") for n in state['joint_names']]
        elif isinstance(state['joint_names'][0], str):
            joint_names = [str(n) for n in state['joint_names']]
        else:
            raise NotImplementedError(type(state['joint_names'][0]))
        joint_state.name = joint_names
        return joint_state

    def dynamics_dataset_metadata(self):
        metadata = super().dynamics_dataset_metadata()
        joint_state: JointState = self.robot._joint_state_listener.get()
        metadata.update({
            'joint_names': joint_state.name,
        })
        return metadata

    @staticmethod
    def simple_name():
        return "dual_arm"

    def get_excluded_models_for_env(self):
        exclude = ExcludeModelsRequest()
        res: ExcludeModelsResponse = self.exclude_from_planning_scene_srv(exclude)
        return res.all_model_names

    def is_motion_feasible(self, environment: Dict, state: Dict, action: Dict):
        joint_state = self.joint_state_msg_from_state_dict(state)
        self.robot.display_robot_state(joint_state, label='check_feasible')

        left_gripper_points = [action['left_gripper_position']]
        right_gripper_points = [action['right_gripper_position']]
        tool_names = [self.robot.left_tool_name, self.robot.right_tool_name]
        grippers = [left_gripper_points, right_gripper_points]

        preferred_tool_orientations = []
        for tool_name in tool_names:
            if 'left' in tool_name:
                preferred_tool_orientations.append(self.left_preferred_tool_orientation)
            elif 'right' in tool_name:
                preferred_tool_orientations.append(self.right_preferred_tool_orientation)
            else:
                raise NotImplementedError()
        # NOTE: we don't use environment here because we assume the planning scenes is static,
        #  so the jacobian follower will already have that information.
        robot_state = RobotState()
        robot_state.joint_state.name = state['joint_names']
        robot_state.joint_state.position = state['joint_positions']
        robot_state.joint_state.velocity = [0.0] * self.get_n_joints()
        plan: RobotTrajectory
        target_reached: bool
        plan, target_reached = self.robot.jacobian_follower.plan_from_start_state(
            start_state=robot_state,
            group_name='both_arms',
            tool_names=tool_names,
            preferred_tool_orientations=preferred_tool_orientations,
            grippers=grippers,
            max_velocity_scaling_factor=0.1,
            max_acceleration_scaling_factor=0.1,
        )

        if len(plan.joint_trajectory.points) == 0:
            predicted_joint_positions = state['joint_positions']
        else:
            final_point: JointTrajectoryPoint = plan.joint_trajectory.points[-1]
            robot_state.joint_state.position = state['joint_positions']
            predicted_joint_positions = []
            for joint_name in state['joint_names']:
                if joint_name in plan.joint_trajectory.joint_names:
                    joint_idx_in_final_point = plan.joint_trajectory.joint_names.index(joint_name)
                    joint_position = final_point.positions[joint_idx_in_final_point]
                elif joint_name in state['joint_names']:
                    joint_idx_in_state = state['joint_names'].index(joint_name)
                    joint_position = float(state['joint_positions'][joint_idx_in_state])
                else:
                    raise ValueError(f"joint {joint_name} is in neither the start state nor the the planed trajectory")
                predicted_joint_positions.append(joint_position)

        return target_reached, predicted_joint_positions

    def initial_obstacle_poses_with_noise(self, env_rng: np.random.RandomState, obstacles: List):
        raise NotImplementedError()

    def get_environment(self, params: Dict, **kwargs):
        default_res = 0.01
        if 'res' not in params:
            rospy.logwarn(f"res not in params, using default {default_res}", logger_name=Path(__file__).stem)
            res = default_res
        else:
            res = params["res"]
        return get_environment_for_extents_3d(extent=params['extent'],
                                              res=res,
                                              service_provider=self.service_provider,
                                              excluded_models=self.get_excluded_models_for_env())

    @staticmethod
    def robot_name():
        raise NotImplementedError()

    def reset_cdcpd(self):
        # since the launch file has respawn=true, we just need to kill cdcpd_node and it will restart
        rosnode.kill_nodes("cdcpd_node")

    def needs_reset(self, state: Dict, params: Dict):
        grippers_out_of_bounds = self.grippers_out_of_bounds(state['left_gripper'], state['right_gripper'], params)
        return super().needs_reset(state, params) or grippers_out_of_bounds
