import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from pyjacobian_follower import IkParams, JacobianFollower

import rosnode
from arm_robots.robot_utils import merge_joint_state_and_scene_msg
from link_bot_data.dataset_utils import add_predicted, deserialize_scene_msg
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from link_bot_pycommon.get_dual_arm_robot_state import GetDualArmRobotState
from link_bot_pycommon.grid_utils import batch_center_res_shape_to_origin_point
from link_bot_pycommon.lazy import Lazy
from link_bot_pycommon.moveit_planning_scene_mixin import MoveitPlanningSceneScenarioMixin
from link_bot_pycommon.moveit_utils import make_joint_state
from link_bot_pycommon.pycommon import densify_points
from merrrt_visualization.rviz_animation_controller import RvizSimpleStepper
from moonshine.geometry import transform_points_3d, xyzrpy_to_matrices, transformation_jacobian, euler_angle_diff
from moonshine.moonshine_utils import numpify, to_list_of_strings
from moveit_msgs.msg import RobotState, RobotTrajectory, PlanningScene
from sdf_tools import utils_3d
from tf import transformations
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
from tf.transformations import quaternion_from_euler

DEBUG_VIZ_STATE_AUG = False


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


class BaseDualArmRopeScenario(FloatingRopeScenario, MoveitPlanningSceneScenarioMixin):
    DISABLE_CDCPD = True
    ROPE_NAMESPACE = 'rope_3d'

    def __init__(self, robot_namespace: str):
        FloatingRopeScenario.__init__(self)
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
        exclude.model_names.append("rope_3d")
        exclude.model_names.append(self.robot_namespace)
        self.exclude_from_planning_scene_srv(exclude)

    def on_before_data_collection(self, params: Dict):
        self.on_before_get_state_or_execute_action()

        # Set the preferred tool orientations
        self.robot.store_tool_orientations({
            self.robot.left_tool_name:  self.left_preferred_tool_orientation,
            self.robot.right_tool_name: self.right_preferred_tool_orientation,
        })

    def get_n_joints(self):
        return len(self.get_joint_names())

    def get_joint_names(self):
        return self.robot.get_joint_names()

    def get_state(self):
        gt_rope_state_vector = self.get_gazebo_rope_state()
        gt_rope_state_vector = np.array(gt_rope_state_vector, np.float32)

        if self.DISABLE_CDCPD:
            cdcpd_rope_state_vector = gt_rope_state_vector
        else:
            cdcpd_rope_state_vector = self.get_cdcpd_state()

        state = {
            'gt_rope': gt_rope_state_vector,
            'rope':    cdcpd_rope_state_vector,
        }
        state.update(self.get_robot_state.get_state())

        left_gripper_to_rope = np.linalg.norm(state['left_gripper'] - state['rope'][0:3])
        right_gripper_to_rope = np.linalg.norm(state['right_gripper'] - state['rope'][-3:])
        if (right_gripper_to_rope > 0.021) or (left_gripper_to_rope > 0.021):
            rospy.logerr(f"state is inconsistent! {left_gripper_to_rope} {right_gripper_to_rope}")
            self.plot_state_rviz(state, label='debugging1')

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
                                                        service_provider=self.service_provider,
                                                        excluded_models=self.get_excluded_models_for_env())

        env = {}
        env.update({k: np.array(v).astype(np.float32) for k, v in voxel_grid_env.items()})
        sdf, sdf_grad = utils_3d.compute_sdf_and_gradient(voxel_grid_env['env'],
                                                          voxel_grid_env['res'],
                                                          voxel_grid_env['origin_point'])
        env['sdf'] = sdf
        env['sdf_grad'] = sdf_grad
        print("Computing SDF and SDF Grad")
        env.update(MoveitPlanningSceneScenarioMixin.get_environment(self))

        return env

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
            reached = np.allclose(desired_tool_position, pred_tool_position, atol=5e-3)
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
                left_gripper_points = [example['left_gripper_position'][b, t]]
                right_gripper_points = [example['right_gripper_position'][b, t]]
                grippers = [left_gripper_points, right_gripper_points]

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

    def sample_object_augmentation_variables(self, batch_size: int, seed: tfp.util.SeedStream):
        # NOTE: lots of hidden hyper-parameters here :(
        zeros = tf.zeros([batch_size, 6], dtype=tf.float32)
        trans_scale = 0.25
        rot_scale = 0.15
        scale = tf.constant([trans_scale, trans_scale, trans_scale, rot_scale, rot_scale, rot_scale], dtype=tf.float32)
        lim = tf.constant([0.5, 0.5, 0.5, np.pi, np.pi, np.pi], dtype=tf.float32)
        distribution = tfp.distributions.TruncatedNormal(zeros, scale=scale, low=-lim, high=lim)
        transformation_params = distribution.sample(seed=seed())

        return transformation_params

    def compute_obj_points(self, inputs: Dict, num_object_interp: int, batch_size: int):
        keys = [add_predicted('rope'), add_predicted('left_gripper'), add_predicted('right_gripper')]

        def _make_points(k, t):
            v = inputs[k][:, t]
            points = tf.reshape(v, [batch_size, -1, 3])
            points = densify_points(batch_size, points)
            return points

        obj_points_0 = {k: _make_points(k, 0) for k in keys}
        obj_points_1 = {k: _make_points(k, 1) for k in keys}

        def _linspace(k):
            return tf.linspace(obj_points_0[k], obj_points_1[k], num_object_interp, axis=1)

        swept_obj_points = tf.concat([_linspace(k) for k in keys], axis=2)

        # TODO: include the robot as an object here?
        # obj_points = tf.concat([robot_points, swept_obj_points], axis=1)
        obj_points = tf.expand_dims(swept_obj_points, axis=1)

        return obj_points

    def apply_object_augmentation_no_ik(self,
                                        m,
                                        to_local_frame,
                                        inputs: Dict,
                                        batch_size,
                                        time,
                                        h: int,
                                        w: int,
                                        c: int,
                                        ):
        """

        Args:
            m: [b, k, 4, 4]
            to_local_frame: [b, 3]  the 1 can also be equal to time
            inputs:
            batch_size:
            time:
            h:
            w:
            c:

        Returns:

        """
        to_local_frame_expanded = to_local_frame[:, None, None]
        m_expanded = m[:, None]

        # apply those to the rope and grippers
        rope_points = inputs[add_predicted('rope')]
        left_gripper_point = inputs[add_predicted('left_gripper')]
        right_gripper_point = inputs[add_predicted('right_gripper')]
        left_gripper_points = tf.expand_dims(left_gripper_point, axis=-2)
        right_gripper_points = tf.expand_dims(right_gripper_point, axis=-2)

        def _transform(m, points, _to_local_frame):
            points_local_frame = points - _to_local_frame
            points_local_frame_aug = transform_points_3d(m, points_local_frame)
            return points_local_frame_aug + _to_local_frame

        # m is expanded to broadcast across batch & num_points dimensions
        rope_points_aug = _transform(m_expanded, rope_points, to_local_frame_expanded)
        left_gripper_points_aug = _transform(m_expanded, left_gripper_points, to_local_frame_expanded)
        right_gripper_points_aug = _transform(m_expanded, right_gripper_points, to_local_frame_expanded)

        # compute the new action
        left_gripper_position = inputs['left_gripper_position']
        right_gripper_position = inputs['right_gripper_position']
        # m is expanded to broadcast across batch dimensions
        left_gripper_position_aug = _transform(m[:, None], left_gripper_position, to_local_frame)
        right_gripper_position_aug = _transform(m[:, None], right_gripper_position, to_local_frame)

        rope_aug = tf.reshape(rope_points_aug, [batch_size, time, -1])
        left_gripper_aug = tf.reshape(left_gripper_points_aug, [batch_size, time, -1])
        right_gripper_aug = tf.reshape(right_gripper_points_aug, [batch_size, time, -1])

        # Now that we've updated the state/action in inputs, compute the local origin point
        state_aug_0 = {
            'left_gripper':  left_gripper_aug[:, 0],
            'right_gripper': right_gripper_aug[:, 0],
            'rope':          rope_aug[:, 0]
        }
        local_center_aug = self.local_environment_center_differentiable(state_aug_0)
        res = inputs['res']
        local_origin_point_aug = batch_center_res_shape_to_origin_point(local_center_aug, res, h, w, c)

        object_aug_update = {
            add_predicted('rope'):          rope_aug,
            add_predicted('left_gripper'):  left_gripper_aug,
            add_predicted('right_gripper'): right_gripper_aug,
            'left_gripper_position':        left_gripper_position_aug,
            'right_gripper_position':       right_gripper_position_aug,
        }

        if DEBUG_VIZ_STATE_AUG:
            stepper = RvizSimpleStepper()
            for b in debug_viz_batch_indices(batch_size):
                env_b = {
                    'env':          inputs['env'][b],
                    'res':          res[b],
                    'extent':       inputs['extent'][b],
                    'origin_point': inputs['origin_point'][b],
                }

                self.plot_environment_rviz(env_b)
                self.debug_viz_state_action(object_aug_update, b, 'aug', color='white')
                stepper.step()
        return object_aug_update, local_origin_point_aug, local_center_aug

    def compute_collision_free_point_ik(self,
                                        default_robot_state,
                                        points,
                                        group_name,
                                        tip_names,
                                        scene_msg,
                                        ik_params):
        return self.robot.j.compute_collision_free_point_ik(default_robot_state,
                                                            points,
                                                            group_name,
                                                            tip_names,
                                                            scene_msg,
                                                            ik_params)

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

    def aug_solve_ik(self,
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

        def _position_tensor_to_point(_positions):
            return Point(*_positions.numpy())

        for b in range(batch_size):
            scene_msg_b = scene_msg[b]

            default_robot_state_b = RobotState()
            default_robot_state_b.joint_state.position = default_robot_positions[b].numpy().tolist()
            default_robot_state_b.joint_state.name = joint_names
            scene_msg_b.robot_state.joint_state.position = default_robot_positions[b].numpy().tolist()
            scene_msg_b.robot_state.joint_state.name = joint_names
            points_b = [_position_tensor_to_point(left_target_position[b]),
                        _position_tensor_to_point(right_target_position[b])]
            robot_state_b = self.compute_collision_free_point_ik(default_robot_state_b,
                                                                 points_b,
                                                                 group_name,
                                                                 tip_names,
                                                                 scene_msg_b,
                                                                 ik_params)

            reached.append(robot_state_b is not None)
            if robot_state_b is None:
                joint_position_b = default_robot_state_b.joint_state.position
            else:
                joint_position_b = robot_state_b.joint_state.position
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

        Returns:

        """
        default_robot_positions = inputs[add_predicted('joint_positions')][:, 0]
        left_gripper_points_aug = inputs_aug[add_predicted('left_gripper')]
        right_gripper_points_aug = inputs_aug[add_predicted('right_gripper')]
        deserialize_scene_msg(inputs_aug)

        scene_msg = inputs_aug['scene_msg']
        joint_names = to_list_of_strings(inputs_aug['joint_names'][0, 0].numpy().tolist())
        joint_positions_aug_, is_ik_valid = self.aug_solve_ik(scene_msg=scene_msg,
                                                              joint_names=joint_names,
                                                              default_robot_positions=default_robot_positions,
                                                              left_target_position=left_gripper_points_aug[:, 0],
                                                              right_target_position=right_gripper_points_aug[:, 0],
                                                              ik_params=ik_params,
                                                              batch_size=batch_size)
        joint_positions_aug_start = joint_positions_aug_

        # then run the jacobian follower to compute the second new position
        tool_names = [self.robot.left_tool_name, self.robot.right_tool_name]
        preferred_tool_orientations = self.get_preferred_tool_orientations(tool_names)
        joint_positions_aug = []
        reached = []
        for b in range(batch_size):
            scene_msg_b = scene_msg[b]
            grippers_end_b = [left_gripper_points_aug[b, 1][None].numpy(), right_gripper_points_aug[b, 1][None].numpy()]
            joint_positions_aug_start_b = joint_positions_aug_start[b]
            start_joint_state_b = JointState(name=joint_names, position=joint_positions_aug_start_b.numpy())
            empty_scene_msg_b, start_robot_state_b = merge_joint_state_and_scene_msg(scene_msg_b,
                                                                                     start_joint_state_b)
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
            if len(planned_to_end_points) > 0:
                end_joint_state_b = planned_to_end_points[-1].positions
            else:
                end_joint_state_b = joint_positions_aug_start_b  # just a filler

            joint_positions_aug_b = tf.stack([joint_positions_aug_start_b, end_joint_state_b])
            joint_positions_aug.append(joint_positions_aug_b)
            reached.append(reached_end_b)
        reached = tf.stack(reached, axis=0)
        joint_positions_aug = tf.stack(joint_positions_aug, axis=0)
        is_ik_valid = tf.cast(tf.logical_and(is_ik_valid, reached), tf.float32)

        return joint_positions_aug, is_ik_valid

    def initial_identity_aug_params(self, batch_size, k_transforms):
        return tf.zeros([batch_size, k_transforms, 6], tf.float32)

    def sample_target_aug_params(self, seed, aug_params, n_samples):
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
        target_q_b = transformations.quaternion_from_euler(*target_euler_b)
        self.tf.send_transform(target_pos_b, target_q_b, f'aug_opt_initial_{obj_i}', frame_id, False)

    def aug_target_pos(self, target):
        return target[:3]

    def transformation_params_to_matrices(self, obj_transforms):
        return xyzrpy_to_matrices(obj_transforms)

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
            'error':                inputs['error'],
            'is_close':             inputs['is_close'],
            'joint_names':          inputs['joint_names'],
            'scene_msg':            inputs['scene_msg'],
            'time':                 inputs['time'],
            add_predicted('stdev'): inputs[add_predicted('stdev')],
        }
