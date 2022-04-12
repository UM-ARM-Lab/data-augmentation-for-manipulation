import warnings
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional

import hjson
import numpy as np
import tensorflow as tf
from pyjacobian_follower import IkParams, JacobianFollower

import rosnode
from arc_utilities.algorithms import nested_dict_update
from arm_robots.robot_utils import merge_joint_state_and_scene_msg
from link_bot_data.dataset_utils import add_predicted
from link_bot_data.rviz_arrow import rviz_arrow
from link_bot_pycommon.get_dual_arm_robot_state import GetDualArmRobotState
from link_bot_pycommon.lazy import Lazy
from link_bot_pycommon.moveit_planning_scene_mixin import MoveitPlanningSceneScenarioMixin
from link_bot_pycommon.moveit_utils import make_joint_state
from moonshine.filepath_tools import load_params
from moonshine.geometry_tf import transformation_jacobian, euler_angle_diff
from moonshine.numpify import numpify
from moonshine.tensorflow_utils import to_list_of_strings
from moonshine.torch_and_tf_utils import remove_batch, add_batch
from moveit_msgs.msg import RobotState, RobotTrajectory, PlanningScene
from tf.transformations import quaternion_from_euler
from trajectory_msgs.msg import JointTrajectoryPoint

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    import moveit_commander
    from arm_robots.get_robot import get_moveit_robot

import rospy
from arc_utilities.listener import Listener
from geometry_msgs.msg import PoseStamped, Point
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.floating_rope_scenario import FloatingRopeScenario
from link_bot_pycommon.get_occupancy import get_environment_for_extents_3d
from arm_gazebo_msgs.srv import ExcludeModels, ExcludeModelsRequest, ExcludeModelsResponse
from rosgraph.names import ns_join
from sensor_msgs.msg import JointState, PointCloud2

rope_key_name = 'rope'


def get_joint_positions_given_state_and_plan(plan: RobotTrajectory, robot_state: RobotState):
    if len(plan.joint_trajectory.points) == 0:
        predicted_joint_positions = robot_state.joint_state.position
    else:
        final_point: JointTrajectoryPoint = plan.joint_trajectory.points[-1]
        predicted_joint_positions = []
        for joint_name in robot_state.joint_state.name:
            if joint_name in plan.joint_trajectory.joint_names:
                joint_idx_in_final_point = plan.joint_trajectory.joint_names.index(joint_name)
                joint_position = final_point.positions[joint_idx_in_final_point]
            elif joint_name in robot_state.joint_state.name:
                joint_idx_in_state = list(robot_state.joint_state.name).index(joint_name)
                joint_position = float(robot_state.joint_state.position[joint_idx_in_state])
            else:
                raise ValueError(f"joint {joint_name} is in neither the start state nor the the planed trajectory")
            predicted_joint_positions.append(joint_position)
    return predicted_joint_positions


def robot_state_msg_from_state_dict(state: Dict):
    robot_state = RobotState()
    robot_state.joint_state = joint_state_msg_from_state_dict(state)
    robot_state.joint_state.velocity = [0.0] * len(robot_state.joint_state.name)
    aco = state.get('attached_collision_objects', None)
    if aco is not None:
        robot_state.attached_collision_objects = aco
    return robot_state


def joint_state_msg_from_state_dict(state: Dict):
    joint_state = JointState(position=state['joint_positions'], name=to_list_of_strings(state['joint_names']))
    joint_state.header.stamp = rospy.Time.now()
    return joint_state


def joint_state_msg_from_state_dict_predicted(state: Dict):
    joint_state = JointState(position=state[add_predicted('joint_positions')],
                             name=to_list_of_strings(state['joint_names']))
    joint_state.header.stamp = rospy.Time.now()
    return joint_state


def to_point_msg(v):
    return Point(x=v[0], y=v[1], z=v[2])


def joint_positions_with_defaults(joint_names, joint_names_subset, joint_positions_subset, default_robot_state):
    end_joint_positions_b = []
    for joint_name in joint_names:
        if joint_name in joint_names_subset:
            joint_index = joint_names_subset.index(joint_name)
            end_position = joint_positions_subset[joint_index]
        else:
            joint_index = default_robot_state.joint_state.name.index(joint_name)
            end_position = default_robot_state.joint_state.position[joint_index]
        end_joint_positions_b.append(end_position)
    return end_joint_positions_b


def joint_positions_in_order(joint_names, robot_state_b):
    """

    Args:
        joint_names: the names in the order you want them
        robot_state_b: the robot state you want the joint positions to come from

    Returns:
        joint positions, in the order of joint_names

    """
    joint_positions = []
    for joint_name in joint_names:
        robot_state_b_joint_idx = robot_state_b.joint_state.name.index(joint_name)
        joint_positions.append(robot_state_b.joint_state.position[robot_state_b_joint_idx])
    return joint_positions


class BaseDualArmRopeScenario(FloatingRopeScenario, MoveitPlanningSceneScenarioMixin):

    def __init__(self, robot_namespace: str, params):
        FloatingRopeScenario.__init__(self, params)
        MoveitPlanningSceneScenarioMixin.__init__(self, robot_namespace)

        self.robot_namespace = robot_namespace

        self.service_provider = BaseServices()
        joint_state_viz_topic = ns_join(self.robot_namespace, "joint_states_viz")
        self.joint_state_viz_pub = rospy.Publisher(joint_state_viz_topic, JointState, queue_size=10)
        self.cdcpd_listener = Lazy(Listener, "cdcpd/output", PointCloud2)

        # NOTE: you may want to override this for your specific robot/scenario
        self.left_preferred_tool_orientation = quaternion_from_euler(np.pi, 0, 0)
        self.right_preferred_tool_orientation = quaternion_from_euler(np.pi, 0, 0)

        self.size_of_box_around_tool_for_planning = 0.05
        exclude_srv_name = ns_join(self.robot_namespace, "exclude_models_from_planning_scene")
        self.exclude_from_planning_scene_srv = rospy.ServiceProxy(exclude_srv_name, ExcludeModels)

        # FIXME: this blocks until the robot is available
        self.robot = Lazy(get_moveit_robot, self.robot_namespace, raise_on_failure=True)

        self.get_robot_state = GetDualArmRobotState(self.robot)

    @cached_property
    def root_link(self):
        return self.robot.robot_commander.get_root_link()

    def add_boxes_around_tools(self):
        # add attached collision object to prevent moveit from smooshing the rope and ends of grippers into obstacles
        self.moveit_scene = moveit_commander.PlanningSceneInterface(ns=self.robot_namespace)
        self.robust_add_to_scene(self.robot.left_tool_name, 'left_tool_box',
                                 self.robot.get_left_gripper_links() + ['end_effector_left', 'left_tool'])
        self.robust_add_to_scene(self.robot.right_tool_name, 'right_tool_box',
                                 self.robot.get_right_gripper_links() + ['end_effector_right', 'right_tool'])

    def robust_add_to_scene(self, link: str, new_object_name: str, touch_links: List[str]):
        pose = PoseStamped()
        pose.header.frame_id = link
        pose.pose.orientation.w = 1.0
        radius = self.size_of_box_around_tool_for_planning
        while True:
            self.moveit_scene.add_sphere(new_object_name, pose, radius=radius)
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

        self.add_boxes_around_tools()

        # Mark the rope as a not-obstacle
        exclude = ExcludeModelsRequest()
        exclude.model_names.append(self.params['rope_name'])
        exclude.model_names.append(self.robot_namespace)
        self.exclude_from_planning_scene_srv(exclude)

        # Set the preferred tool orientations
        self.robot.store_tool_orientations({
            self.robot.left_tool_name:  self.left_preferred_tool_orientation,
            self.robot.right_tool_name: self.right_preferred_tool_orientation,
        })

    def on_before_data_collection(self, params: Dict):
        self.on_before_get_state_or_execute_action()

    def get_n_joints(self):
        return len(self.get_joint_names())

    def get_joint_names(self):
        return self.robot.get_joint_names()

    def get_state(self):
        gt_rope_state_vector = self.get_gazebo_rope_state()
        gt_rope_state_vector = np.array(gt_rope_state_vector, np.float32)

        state = {
            'rope': gt_rope_state_vector,
        }
        state.update(self.get_robot_state.get_state())

        left_gripper_to_rope = np.linalg.norm(state['left_gripper'] - state['rope'][0:3])
        right_gripper_to_rope = np.linalg.norm(state['right_gripper'] - state['rope'][-3:])

        return state

    def plot_state_rviz(self, state: Dict, **kwargs):
        FloatingRopeScenario.plot_state_rviz(self, state, **kwargs)
        label = kwargs.pop("label", "")
        # FIXME: the ACOs are part of the "environment", but they are needed to plot the state. leaky abstraction :(
        #  perhaps make them part of state_metadata?
        aco = state.get('attached_collision_objects', None)

        if 'joint_positions' in state and 'joint_names' in state:
            joint_state = joint_state_msg_from_state_dict(state)
            robot_state = RobotState(joint_state=joint_state, attached_collision_objects=aco)
            self.robot.display_robot_state(robot_state, label, kwargs.get("color", None))
        if add_predicted('joint_positions') in state and 'joint_names' in state:
            joint_state = joint_state_msg_from_state_dict_predicted(state)
            robot_state = RobotState(joint_state=joint_state, attached_collision_objects=aco)
            self.robot.display_robot_state(robot_state, label, kwargs.get("color", None))
        elif 'joint_positions' not in state:
            rospy.logwarn_throttle(10, 'no joint positions in state', logger_name=Path(__file__).stem)
        elif 'joint_names' not in state:
            rospy.logwarn_throttle(10, 'no joint names in state', logger_name=Path(__file__).stem)

    def dynamics_dataset_metadata(self):
        metadata = FloatingRopeScenario.dynamics_dataset_metadata(self)
        return metadata

    @staticmethod
    def simple_name():
        return "dual_arm"

    def get_excluded_models_for_env(self):
        exclude = ExcludeModelsRequest()
        res: ExcludeModelsResponse = self.exclude_from_planning_scene_srv(exclude)
        return res.all_model_names

    def initial_obstacle_poses_with_noise(self, env_rng: np.random.RandomState, obstacles: List):
        raise NotImplementedError()

    def get_environment(self, params: Dict, **kwargs):
        default_res = 0.02
        if 'res' not in params:
            rospy.logwarn(f"res not in params, using default {default_res}", logger_name=Path(__file__).stem)
            res = default_res
        else:
            res = params["res"]

        voxel_grid_env = get_environment_for_extents_3d(extent=params['extent'],
                                                        res=res,
                                                        frame='robot_root',
                                                        service_provider=self.service_provider,
                                                        excluded_models=self.get_excluded_models_for_env())

        env = {}
        env.update({k: np.array(v).astype(np.float32) for k, v in voxel_grid_env.items()})
        from moonshine.tfa_sdf import compute_sdf_and_gradient_batch
        sdf, sdf_grad = remove_batch(*compute_sdf_and_gradient_batch(*add_batch(voxel_grid_env['env'],
                                                                                voxel_grid_env['res'])))
        sdf = numpify(sdf)
        sdf_grad = numpify(sdf_grad)
        env['sdf'] = sdf
        env['sdf_grad'] = sdf_grad

        env.update(MoveitPlanningSceneScenarioMixin.get_environment(self))

        return env

    def base_link_frame(self):
        return self.robot.robot_commander.get_root_link()

    @staticmethod
    def robot_name():
        raise NotImplementedError()

    def reset_cdcpd(self):
        # since the launch file has respawn=true, we just need to kill cdcpd_node and it will restart
        rosnode.kill_nodes("cdcpd_node")

    def needs_reset(self, state: Dict, params: Dict):
        grippers_out_of_bounds = self.grippers_out_of_bounds(state['left_gripper'], state['right_gripper'], params)
        return FloatingRopeScenario.needs_reset(self, state, params) or grippers_out_of_bounds

    def get_preferred_tool_orientations(self, tool_names: List[str]):
        """
        The purpose of this function it to make sure the tool orientations are in the order of tool_names
        Args:
            tool_names:

        Returns:

        """
        preferred_tool_orientations = []
        for tool_name in tool_names:
            if 'left' in tool_name:
                preferred_tool_orientations.append(self.left_preferred_tool_orientation)
            elif 'right' in tool_name:
                preferred_tool_orientations.append(self.right_preferred_tool_orientation)
            else:
                raise NotImplementedError()
        return preferred_tool_orientations

    def is_moveit_robot_in_collision(self, environment: Dict, state: Dict, action: Dict):
        scene: PlanningScene = environment['scene_msg']
        joint_state = joint_state_msg_from_state_dict(state)
        scene, robot_state = merge_joint_state_and_scene_msg(scene, joint_state)
        in_collision = self.robot.jacobian_follower.check_collision(scene, robot_state)
        return in_collision

    def moveit_robot_reached(self, state: Dict, action: Dict, next_state: Dict):
        tool_names = [self.robot.left_tool_name, self.robot.right_tool_name]
        predicted_robot_state = robot_state_msg_from_state_dict(next_state)
        desired_tool_positions = [action['left_gripper_position'], action['right_gripper_position']]
        pred_tool_positions = self.robot.jacobian_follower.get_tool_positions(tool_names, predicted_robot_state)
        for pred_tool_position, desired_tool_position in zip(pred_tool_positions, desired_tool_positions):
            desired_tool_position_root = desired_tool_position
            reached = np.allclose(desired_tool_position_root, pred_tool_position, atol=5e-3)
            if not reached:
                return False
        return True

    def follow_jacobian_from_example(self, example: Dict, j: Optional[JacobianFollower] = None):
        if j is None:
            j = self.robot.jacobian_follower
        batch_size = example["batch_size"]
        scene_msg = example['scene_msg']
        tool_names = [self.robot.left_tool_name, self.robot.right_tool_name]
        preferred_tool_orientations = self.get_preferred_tool_orientations(tool_names)
        target_reached_batched = []
        pred_joint_positions_batched = []
        joint_names_batched = []
        for b in range(batch_size):
            scene_msg_b: PlanningScene = scene_msg[b]
            input_sequence_length = example['left_gripper_position'].shape[1]
            target_reached = [True]
            pred_joint_positions = [example['joint_positions'][b, 0]]
            pred_joint_positions_t = example['joint_positions'][b, 0]
            joint_names_t = example['joint_names'][b, 0]
            joint_names = [joint_names_t]
            for t in range(input_sequence_length):
                # Transform into the right frame
                left_gripper_point = example['left_gripper_position'][b, t]
                right_gripper_point = example['right_gripper_position'][b, t]
                grippers = [[left_gripper_point], [right_gripper_point]]

                joint_state_b_t = make_joint_state(pred_joint_positions_t, to_list_of_strings(joint_names_t))
                scene_msg_b, robot_state = merge_joint_state_and_scene_msg(scene_msg_b, joint_state_b_t)
                plan: RobotTrajectory
                reached_t: bool
                plan, reached_t = j.plan(group_name='both_arms',
                                         tool_names=tool_names,
                                         preferred_tool_orientations=preferred_tool_orientations,
                                         start_state=robot_state,
                                         scene=scene_msg_b,
                                         grippers=grippers,
                                         max_velocity_scaling_factor=0.1,
                                         max_acceleration_scaling_factor=0.1)
                pred_joint_positions_t = get_joint_positions_given_state_and_plan(plan, robot_state)

                target_reached.append(reached_t)
                pred_joint_positions.append(pred_joint_positions_t)
                joint_names.append(joint_names_t)
            target_reached_batched.append(target_reached)
            pred_joint_positions_batched.append(pred_joint_positions)
            joint_names_batched.append(joint_names)

        pred_joint_positions_batched = np.array(pred_joint_positions_batched)
        target_reached_batched = np.array(target_reached_batched)
        joint_names_batched = np.array(joint_names_batched)
        return target_reached_batched, pred_joint_positions_batched, joint_names_batched

    def sample_object_augmentation_variables(self, batch_size: int, seed):
        import tensorflow_probability as tfp
        # NOTE: lots of hidden hyper-parameters here :(
        zeros = tf.zeros([batch_size, 6], dtype=tf.float32)
        trans_scale = 0.25
        rot_scale = 0.15
        scale = tf.constant([trans_scale, trans_scale, trans_scale, rot_scale, rot_scale, rot_scale], dtype=tf.float32)
        lim = tf.constant([0.5, 0.5, 0.5, np.pi, np.pi, np.pi], dtype=tf.float32)
        distribution = tfp.distributions.TruncatedNormal(zeros, scale=scale, low=-lim, high=lim)
        transformation_params = distribution.sample(seed=seed())

        return transformation_params

    def debug_viz_state_action(self, inputs, b, label: str, color='red'):
        state_keys = ['left_gripper', 'right_gripper', 'rope', 'joint_positions']
        action_keys = ['left_gripper_position', 'right_gripper_position']
        state_0 = numpify({k: inputs[add_predicted(k)][b, 0] for k in state_keys})
        state_0['joint_names'] = inputs['joint_names'][b, 0]
        action_0 = numpify({k: inputs[k][b, 0] for k in action_keys})
        state_1 = numpify({k: inputs[add_predicted(k)][b, 1] for k in state_keys})
        state_1['joint_names'] = inputs['joint_names'][b, 1]
        self.plot_state_rviz(state_0, idx=0, label=label, color=color)
        self.plot_state_rviz(state_1, idx=1, label=label, color=color)
        self.plot_action_rviz(state_0, action_0, idx=1, label=label, color=color)
        if 'is_close' in inputs:
            self.plot_is_close(inputs['is_close'][b, 1])
        if 'error' in inputs:
            error_t = inputs['error'][b, 1]
            self.plot_error_rviz(error_t)

    def aug_ik_to_start(self,
                        scene_msg: List[PlanningScene],
                        joint_names,
                        default_robot_positions,
                        batch_size: int,
                        left_target_position,
                        right_target_position,
                        ik_params: IkParams,
                        group_name='both_arms',
                        tip_names=None,
                        ):
        if tip_names is None:
            tip_names = ['left_tool', 'right_tool']
        robot_state_b: RobotState
        joint_positions = []
        reached = []

        for b in range(batch_size):
            scene_msg_b = scene_msg[b]

            default_robot_state_b = RobotState()
            default_robot_state_b.joint_state.position = default_robot_positions[b].numpy().tolist()
            default_robot_state_b.joint_state.name = joint_names
            scene_msg_b.robot_state.joint_state.position = default_robot_positions[b].numpy().tolist()
            scene_msg_b.robot_state.joint_state.name = joint_names

            left_target_position_b = left_target_position[b]
            right_target_position_b = right_target_position[b]
            points_b = [Point(*left_target_position_b), Point(*right_target_position_b)]

            robot_state_b = self.robot.jacobian_follower.compute_collision_free_point_ik(default_robot_state_b,
                                                                                         points_b,
                                                                                         group_name,
                                                                                         tip_names,
                                                                                         scene_msg_b,
                                                                                         ik_params)

            reached.append(robot_state_b is not None)
            if robot_state_b is None:
                joint_position_b = default_robot_state_b.joint_state.position
            else:
                joint_position_b = joint_positions_in_order(joint_names, robot_state_b)
            joint_positions.append(tf.convert_to_tensor(joint_position_b, dtype=tf.float32))

        joint_positions = tf.stack(joint_positions, axis=0)
        reached = tf.stack(reached, axis=0)
        return joint_positions, reached

    def aug_ik(self,
               inputs: Dict,
               inputs_aug: Dict,
               ik_params: IkParams,
               batch_size: int):
        """

        Args:
            inputs:
            inputs_aug: a dict containing the desired gripper positions as well as the scene_msg and other state info
            batch_size:

        Returns: [b], keys

        """
        from link_bot_data.tf_dataset_utils import deserialize_scene_msg
        default_robot_positions = inputs[add_predicted('joint_positions')][:, 0]
        left_gripper_points_aug = inputs_aug[add_predicted('left_gripper')]
        right_gripper_points_aug = inputs_aug[add_predicted('right_gripper')]
        deserialize_scene_msg(inputs_aug)

        scene_msg = inputs_aug['scene_msg']
        joint_names = to_list_of_strings(inputs_aug['joint_names'][0, 0].numpy().tolist())
        joint_positions_aug_, is_ik_valid = self.aug_ik_to_start(scene_msg=scene_msg,
                                                                 joint_names=joint_names,
                                                                 default_robot_positions=default_robot_positions,
                                                                 left_target_position=left_gripper_points_aug[:, 0],
                                                                 right_target_position=right_gripper_points_aug[:, 0],
                                                                 ik_params=ik_params,
                                                                 batch_size=batch_size)
        joint_positions_aug_start = joint_positions_aug_  # [b, n_joints]

        # then run the jacobian follower to compute the second new position
        tool_names = [self.robot.left_tool_name, self.robot.right_tool_name]
        preferred_tool_orientations = self.get_preferred_tool_orientations(tool_names)
        joint_positions_aug = []
        reached = []
        for b in range(batch_size):
            scene_msg_b = scene_msg[b]
            joint_positions_aug_start_b = joint_positions_aug_start[b]
            start_joint_state_b = JointState(name=joint_names, position=joint_positions_aug_start_b.numpy())
            empty_scene_msg_b, start_robot_state_b = merge_joint_state_and_scene_msg(scene_msg_b,
                                                                                     start_joint_state_b)

            left_gripper_point_aug_b_end = left_gripper_points_aug[b, 1]
            right_gripper_point_aug_b_end = right_gripper_points_aug[b, 1]
            grippers_end_b = [[left_gripper_point_aug_b_end], [right_gripper_point_aug_b_end]]

            plan_to_end, reached_end_b = self.robot.jacobian_follower.plan(
                group_name='whole_body',
                tool_names=tool_names,
                preferred_tool_orientations=preferred_tool_orientations,
                start_state=start_robot_state_b,
                scene=scene_msg_b,
                grippers=grippers_end_b,
                max_velocity_scaling_factor=0.1,
                max_acceleration_scaling_factor=0.1,
            )
            planned_to_end_points = plan_to_end.joint_trajectory.points
            planned_joint_names = plan_to_end.joint_trajectory.joint_names
            if len(planned_to_end_points) > 0:
                planned_joint_positions = planned_to_end_points[-1].positions
                end_joint_positions_b = joint_positions_with_defaults(joint_names=joint_names,
                                                                      joint_names_subset=planned_joint_names,
                                                                      joint_positions_subset=planned_joint_positions,
                                                                      default_robot_state=start_robot_state_b)
            else:
                end_joint_positions_b = joint_positions_aug_start_b  # just a filler

            joint_positions_aug_b = tf.stack([joint_positions_aug_start_b, end_joint_positions_b])
            joint_positions_aug.append(joint_positions_aug_b)
            reached.append(reached_end_b)
        reached = tf.stack(reached, axis=0)
        joint_positions_aug = tf.stack(joint_positions_aug, axis=0)
        is_ik_valid = tf.cast(tf.logical_and(is_ik_valid, reached), tf.float32)

        joints_pos_k = add_predicted('joint_positions')
        inputs_aug.update({
            joints_pos_k: joint_positions_aug,
        })
        return is_ik_valid, [joints_pos_k]

    def initial_identity_aug_params(self, batch_size, k_transforms):
        return tf.zeros([batch_size, k_transforms, 6], tf.float32)

    def sample_target_aug_params(self, seed, aug_params, n_samples):
        import tensorflow_probability as tfp
        trans_lim = tf.ones([3]) * aug_params['target_trans_lim']
        trans_distribution = tfp.distributions.Uniform(low=-trans_lim, high=trans_lim)

        # NOTE: by restricting the sample of euler angles to < pi/2 we can ensure that the representation is unique.
        #  (see https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf) which allows us to use a simple distance euler
        #  function between two sets of euler angles.
        euler_lim = tf.ones([3]) * aug_params['target_euler_lim']
        euler_distribution = tfp.distributions.Uniform(low=-euler_lim, high=euler_lim)

        trans_target = trans_distribution.sample(sample_shape=n_samples, seed=seed())
        euler_target = euler_distribution.sample(sample_shape=n_samples, seed=seed())

        target_params = tf.concat([trans_target, euler_target], 1)
        return target_params

    def plot_transform(self, obj_i, transform_params, frame_id):
        """

        Args:
            frame_id:
            transform_params: [x,y,z,roll,pitch,yaw]

        Returns:

        """
        target_pos_b = transform_params[:3].numpy()
        target_euler_b = transform_params[3:].numpy()
        target_q_b = quaternion_from_euler(*target_euler_b)
        self.tf.send_transform(target_pos_b, target_q_b, f'aug_opt_initial_{obj_i}', frame_id, False)

    def aug_target_pos(self, target):
        return target[:3]

    def aug_transformation_jacobian(self, obj_transforms):
        return transformation_jacobian(obj_transforms)

    def aug_distance(self, transforms1, transforms2):
        trans1 = transforms1[..., :3]
        trans2 = transforms2[..., :3]
        euler1 = transforms1[..., 3:]
        euler2 = transforms2[..., 3:]
        euler_dist = tf.linalg.norm(euler_angle_diff(euler1, euler2), axis=-1)
        trans_dist = tf.linalg.norm(trans1 - trans2, axis=-1)
        distances = trans_dist + euler_dist
        max_distance = tf.reduce_max(distances)
        return max_distance

    @staticmethod
    def aug_copy_inputs(inputs):
        return {
            'metadata':             inputs['metadata'],
            'error':                inputs['error'],
            'is_close':             inputs['is_close'],
            'joint_names':          inputs['joint_names'],
            'scene_msg':            inputs['scene_msg'],
            'time':                 inputs['time'],
            add_predicted('stdev'): inputs[add_predicted('stdev')],
        }

    @staticmethod
    def aug_merge_hparams(dataset_dir, out_example, outdir):
        in_hparams = load_params(dataset_dir)
        update = {
            'used_augmentation': True,
        }
        out_hparams = deepcopy(in_hparams)
        nested_dict_update(out_hparams, update)
        with (outdir / 'hparams.hjson').open("w") as out_f:
            hjson.dump(out_hparams, out_f)

    def aug_plot_dir_arrow(self, target_pos, scale, frame_id, k):
        dir_msg = rviz_arrow([0, 0, 0], target_pos, scale=scale)
        dir_msg.header.frame_id = frame_id
        dir_msg.id = k
        self.aug_dir_pub.publish(dir_msg)

    def simple_noise(self, rng: np.random.RandomState, example, k: str, v, noise_params):
        mean = 0
        if k in noise_params:
            std = noise_params[k]
            noise = rng.randn(*v.shape) * std + mean
            v_out = v + noise
        else:
            v_out = v
        return v_out
