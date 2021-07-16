import pathlib
from dataclasses import dataclass
from typing import Dict, List

import tensorflow as tf
import tensorflow_probability as tfp
import transformations

import rospy
from arm_robots.robot import MoveitEnabledRobot
from arm_robots.robot_utils import merge_joint_state_and_scene_msg
from geometry_msgs.msg import Point
from learn_invariance.invariance_model_wrapper import InvarianceModelWrapper
from link_bot_classifiers.classifier_debugging import ClassifierDebugging
from link_bot_classifiers.local_env_helper import LocalEnvHelper
from link_bot_classifiers.make_voxelgrid_inputs import VoxelgridInfo
from link_bot_data.dataset_utils import add_new, add_predicted, deserialize_scene_msg
from link_bot_pycommon.bbox_visualization import grid_to_bbox
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from link_bot_pycommon.grid_utils import lookup_points_in_vg, send_voxelgrid_tf_origin_point_res, environment_to_vg_msg, \
    occupied_voxels_to_points, subtract, binary_or
from link_bot_pycommon.pycommon import densify_points
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from merrrt_visualization.rviz_animation_controller import RvizSimpleStepper
from moonshine.geometry import transform_points_3d, pairwise_squared_distances, transformation_params_to_matrices
from moonshine.moonshine_utils import reduce_mean_no_nan, repeat, to_list_of_strings, possibly_none_concat
from moonshine.raster_3d import points_to_voxel_grid_res_origin_point
from moveit_msgs.msg import RobotState, PlanningScene
from sensor_msgs.msg import JointState


def debug_aug():
    return rospy.get_param("DEBUG_AUG", False)


def debug_aug_sgd():
    return rospy.get_param("DEBUG_AUG_SGD", False)


def debug_ik():
    return rospy.get_param("DEBUG_IK", False)


@dataclass
class MinDists:
    attract: tf.Tensor
    repel: tf.Tensor
    robot_repel: tf.Tensor


@dataclass
class EnvPoints:
    full: tf.Tensor
    sparse: tf.Tensor


@dataclass
class EnvOptDebugVars:
    nearest_attract_env_points: tf.Tensor
    nearest_repel_points: tf.Tensor
    nearest_robot_repel_points: tf.Tensor


def subsample_points(points, fraction):
    """

    Args:
        points: [n, 3]
        fraction: from 0.0 to 1.0

    Returns:

    """
    n_take_every = int(1 / fraction)
    return points[::n_take_every]


class BioIKSolver:

    def __init__(self, robot: MoveitEnabledRobot, group_name='both_arms', position_only=True):
        self.position_only = position_only
        self.robot = robot
        self.group_name = group_name
        self.j = robot.jacobian_follower
        if self.group_name == 'both_arms':
            self.tip_names = ['left_tool', 'right_tool']
        else:
            raise NotImplementedError()

    def solve(self, scene_msg: List[PlanningScene],
              joint_names,
              default_robot_positions,
              batch_size: int,
              left_target_position=None,
              right_target_position=None,
              left_target_pose=None,
              right_target_pose=None):
        """

        Args:
            scene_msg: [b] list of PlanningScene
            left_target_position:  [b,3], xyz
            right_target_position: [b,3], xyz
            left_target_pose:  [b,7] euler xyz, quat xyzw
            right_target_pose: [b,7] euler xyz, quat xyzw

        Returns:

        """
        robot_state_b: RobotState
        joint_positions = []
        reached = []

        def _pose_tensor_to_point(_positions):
            raise NotImplementedError()

        def _position_tensor_to_point(_positions):
            return Point(*_positions.numpy())

        for b in range(batch_size):
            scene_msg_b = scene_msg[b]

            default_robot_state_b = RobotState()
            default_robot_state_b.joint_state.position = default_robot_positions[b].numpy().tolist()
            default_robot_state_b.joint_state.name = joint_names
            scene_msg_b.robot_state.joint_state.position = default_robot_positions[b].numpy().tolist()
            scene_msg_b.robot_state.joint_state.name = joint_names
            if self.position_only:
                points_b = [_position_tensor_to_point(left_target_position[b]),
                            _position_tensor_to_point(right_target_position[b])]
                robot_state_b = self.j.compute_collision_free_point_ik(default_robot_state_b, points_b, self.group_name,
                                                                       self.tip_names,
                                                                       scene_msg_b)
            else:
                poses_b = [left_target_pose[b], right_target_pose[b]]
                robot_state_b = self.j.compute_collision_free_pose_ik(default_robot_state_b, poses_b, self.group_name,
                                                                      self.tip_names, scene_msg_b)

            reached.append(robot_state_b is not None)
            if robot_state_b is None:
                joint_position_b = default_robot_state_b.joint_state.position
            else:
                joint_position_b = robot_state_b.joint_state.position
            joint_positions.append(tf.convert_to_tensor(joint_position_b, dtype=tf.float32))

        joint_positions = tf.stack(joint_positions, axis=0)
        reached = tf.stack(reached, axis=0)
        return joint_positions, reached


class AugmentationOptimization:

    def __init__(self,
                 scenario: ScenarioWithVisualization,
                 debug: ClassifierDebugging,
                 local_env_helper: LocalEnvHelper,
                 vg_info: VoxelgridInfo,
                 points_state_keys: List[str],
                 hparams: Dict,
                 batch_size: int,
                 state_keys: List[str],
                 action_keys: List[str]):
        self.state_keys = state_keys
        self.action_keys = action_keys
        self.hparams = hparams.get('augmentation', None)
        self.points_state_keys = points_state_keys
        self.batch_size = batch_size
        self.scenario = scenario
        self.vg_info = vg_info
        self.debug = debug
        self.local_env_helper = local_env_helper
        self.broadcaster = self.scenario.tf.tf_broadcaster
        self.ik_solver = BioIKSolver(scenario.robot)

        self.robot_subsample = 0.5
        self.env_subsample = 0.25
        self.num_object_interp = 5  # must be >=2
        self.num_robot_interp = 3  # must be >=2
        self.max_steps = 25
        self.gen = tf.random.Generator.from_seed(0)
        self.seed = tfp.util.SeedStream(1, salt="nn_classifier_aug")
        self.step_size = 10.0
        self.opt = tf.keras.optimizers.SGD(self.step_size)
        self.step_size_threshold = 0.001  # stopping criteria, how far the env moved (meters)
        self.barrier_upper_lim = tf.square(0.06)  # stops repelling points from pushing after this distance
        self.barrier_scale = 0.05  # scales the gradients for the repelling points
        self.grad_clip = 0.25  # max dist step the env aug update can take
        self.attract_weight = 0.2
        self.repel_weight = 1.0
        self.invariance_weight = 0.01
        self.ground_penetration_weight = 1.0
        self.robot_base_penetration_weight = 1.0

        # Precompute this for speed
        self.barrier_epsilon = 0.01
        self.log_cutoff = tf.math.log(self.barrier_scale * self.barrier_upper_lim + self.barrier_epsilon)

        if self.hparams is not None:
            invariance_model_path = pathlib.Path(self.hparams['invariance_model'])
            self.invariance_model_wrapper = InvarianceModelWrapper(invariance_model_path, self.batch_size,
                                                                   self.scenario)

        # metrics
        self.is_valids = None
        self.local_env_aug_fix_delta = None

    def augmentation_optimization(self,
                                  inputs: Dict,
                                  batch_size,
                                  time):
        aug_type = self.hparams['type']
        if aug_type == 'optimization':
            return self.augmentation_optimization1(inputs, batch_size, time)
        elif aug_type == 'optimization2':
            return self.augmentation_optimization2(inputs, batch_size, time)
        else:
            raise NotImplementedError(aug_type)

    def augmentation_optimization2(self,
                                   inputs: Dict,
                                   batch_size,
                                   time):
        _setup = self.setup(inputs, batch_size)
        inputs_aug, res, object_points, object_points_occupancy, local_env, local_origin_point = _setup

        new_env = self.get_new_env(inputs)
        # Augment every example in the batch with the _same_ new environment. This is faster/less memory then doing it
        # independently, but shouldn't really change the overall effect because each batch will have a randomly selected
        # new environment. Furthermore, in most cases we test with only one new environment, in which case this is
        # actually identical.
        new_env_0 = {k: v[0] for k, v in new_env.items()}
        inputs_aug, local_origin_point_aug, local_center_aug, local_env_aug = self.opt_object_augmentation(inputs,
                                                                                                           inputs_aug,
                                                                                                           new_env_0,
                                                                                                           object_points,
                                                                                                           object_points_occupancy,
                                                                                                           res,
                                                                                                           batch_size,
                                                                                                           time)
        joint_positions_aug, is_ik_valid = self.solve_ik(inputs, inputs_aug, new_env, batch_size)
        inputs_aug.update({
            add_predicted('joint_positions'): joint_positions_aug,
            'joint_names':                    inputs['joint_names'],
        })

        is_valid = is_ik_valid
        self.is_valids = possibly_none_concat(self.is_valids, is_valid, axis=0)

        if debug_aug():
            stepper = RvizSimpleStepper()
            for b in debug_viz_batch_indices(batch_size):
                print(bool(is_valid[b].numpy()))
                _aug_dict = {
                    'env':          local_env_aug[b].numpy(),
                    'origin_point': local_origin_point_aug[b].numpy(),
                    'res':          res[b].numpy(),
                }
                msg = environment_to_vg_msg(_aug_dict, frame='local_env_aug_vg', stamp=rospy.Time(0))
                self.debug.env_aug_pub5.publish(msg)
                send_voxelgrid_tf_origin_point_res(self.broadcaster,
                                                   local_origin_point_aug[b],
                                                   res[b],
                                                   frame='local_env_aug_vg')

                self.debug.plot_state_rviz(inputs_aug, b, 0, 'aug', color='blue')
                self.debug.plot_state_rviz(inputs_aug, b, 1, 'aug', color='blue')
                self.debug.plot_action_rviz(inputs_aug, b, 'aug', color='blue')
                # stepper.step()  # FINAL AUG (not necessarily what the network sees, only if valid)
                # print(env_aug_valid[b], object_aug_valid[b])

        # FIXME: this is so hacky
        keys_aug = [add_predicted('joint_positions')]
        keys_aug += self.action_keys
        keys_aug += [add_predicted(psk) for psk in self.points_state_keys]
        for k in keys_aug:
            v = inputs_aug[k]
            iv = tf.reshape(is_valid, [batch_size] + [1] * (v.ndim - 1))
            inputs_aug[k] = iv * inputs_aug[k] + (1 - iv) * inputs[k]

        iv = tf.reshape(is_valid, [batch_size] + [1, 1, 1])
        local_env_aug = iv * local_env_aug + (1 - iv) * local_env

        iv = tf.reshape(is_valid, [batch_size] + [1])
        local_origin_point_aug = iv * local_origin_point_aug + (1 - iv) * local_origin_point

        return inputs_aug, local_env_aug, local_origin_point_aug

    def augmentation_optimization1(self,
                                   inputs: Dict,
                                   batch_size,
                                   time):
        _setup = self.setup(inputs, batch_size)
        inputs_aug, res, object_points, object_points_occupancy, local_env, local_origin_point = _setup

        object_transforms = self.sample_object_transformations(batch_size)
        object_aug_valid, object_aug_update, local_origin_point_aug = self.apply_object_augmentation(object_transforms,
                                                                                                     inputs,
                                                                                                     batch_size,
                                                                                                     time)
        inputs_aug.update(object_aug_update)

        # this was just updated by apply_state_augmentation
        if debug_aug():
            stepper = RvizSimpleStepper()
            for b in debug_viz_batch_indices(batch_size):
                self.debug.send_position_transform(local_origin_point_aug[b], 'local_origin_point_aug')
                debug_i = tf.squeeze(tf.where(1 - object_points_occupancy[b]), -1)
                points_debug_b = tf.gather(object_points[b], debug_i)
                self.scenario.plot_points_rviz(points_debug_b.numpy(), label='repel', color='r')

                debug_i = tf.squeeze(tf.where(object_points_occupancy[b]), -1)
                points_debug_b = tf.gather(object_points[b], debug_i)
                self.scenario.plot_points_rviz(points_debug_b.numpy(), label='attract', color='g')

                send_voxelgrid_tf_origin_point_res(self.broadcaster,
                                                   origin_point=local_origin_point_aug[b],
                                                   res=res[b],
                                                   frame='local_env_aug_vg')

                bbox_msg = grid_to_bbox(rows=self.local_env_helper.h,
                                        cols=self.local_env_helper.w,
                                        channels=self.local_env_helper.c,
                                        resolution=res[b].numpy())
                bbox_msg.header.frame_id = 'local_env_aug_vg'

                self.debug.aug_bbox_pub.publish(bbox_msg)
                # stepper.step()

        object_points_aug = transform_points_3d(object_transforms[:, None], object_points)
        robot_points_aug = self.compute_swept_robot_points(inputs_aug, batch_size)

        if debug_aug():
            for b in debug_viz_batch_indices(batch_size):
                debug_i = tf.squeeze(tf.where(object_points_occupancy[b]), -1)
                points_debug_b = tf.gather(object_points_aug[b], debug_i)
                self.scenario.plot_points_rviz(points_debug_b.numpy(), label='attract_aug', color='g', scale=0.005)

                debug_i = tf.squeeze(tf.where(1 - object_points_occupancy[b]), -1)
                points_debug_b = tf.gather(object_points_aug[b], debug_i)
                self.scenario.plot_points_rviz(points_debug_b.numpy(), label='repel_aug', color='r', scale=0.005)

                robot_points_aug_b = robot_points_aug[b]
                self.scenario.plot_points_rviz(robot_points_aug_b.numpy(), label='robot_aug', color='m', scale=0.005)
                # stepper.step()

        new_env = self.get_new_env(inputs)
        env_aug_valid, local_env_aug = self.opt_new_env_augmentation(inputs_aug,
                                                                     new_env,
                                                                     object_points_aug,
                                                                     robot_points_aug,
                                                                     object_points_occupancy,
                                                                     None,
                                                                     res,
                                                                     local_origin_point_aug,
                                                                     batch_size)

        if debug_aug():
            stepper = RvizSimpleStepper()
            for b in debug_viz_batch_indices(batch_size):
                _aug_dict = {
                    'env':          local_env_aug[b].numpy(),
                    'origin_point': local_origin_point_aug[b].numpy(),
                    'res':          res[b].numpy(),
                }
                msg = environment_to_vg_msg(_aug_dict, frame='local_env_aug_vg', stamp=rospy.Time(0))
                self.debug.env_aug_pub5.publish(msg)
                send_voxelgrid_tf_origin_point_res(self.broadcaster,
                                                   local_origin_point_aug[b],
                                                   res[b],
                                                   frame='local_env_aug_vg')

                self.debug.plot_state_rviz(inputs_aug, b, 0, 'aug', color='blue')
                self.debug.plot_state_rviz(inputs_aug, b, 1, 'aug', color='blue')
                self.debug.plot_action_rviz(inputs_aug, b, 'aug', color='blue')
                # stepper.step()  # FINAL AUG (not necessarily what the network sees, only if valid)

                print(env_aug_valid[b], object_aug_valid[b])

        is_valid = env_aug_valid * object_aug_valid
        self.is_valids = tf.concat([self.is_valids, is_valid], axis=0)

        # FIXME: this is so hacky
        keys_aug = [add_predicted('joint_positions')]
        keys_aug += self.action_keys
        keys_aug += [add_predicted(psk) for psk in self.points_state_keys]
        for k in keys_aug:
            v = inputs_aug[k]
            iv = tf.reshape(is_valid, [batch_size] + [1] * (v.ndim - 1))
            inputs_aug[k] = iv * inputs_aug[k] + (1 - iv) * inputs[k]

        iv = tf.reshape(is_valid, [batch_size] + [1, 1, 1])
        local_env_aug = iv * local_env_aug + (1 - iv) * local_env

        iv = tf.reshape(is_valid, [batch_size] + [1])
        local_origin_point_aug = iv * local_origin_point_aug + (1 - iv) * local_origin_point

        return inputs_aug, local_env_aug, local_origin_point_aug

    def setup(self, inputs, batch_size):
        inputs_aug = {
            # initialize with things that we won't be updating in this augmentation
            'res':                  inputs['res'],
            'extent':               inputs['extent'],
            'origin_point':         inputs['origin_point'],
            'env':                  inputs['env'],
            'is_close':             inputs['is_close'],
            'batch_size':           inputs['batch_size'],
            'time':                 inputs['time'],
            add_predicted('stdev'): inputs[add_predicted('stdev')],
        }

        local_env, local_origin_point = self.get_local_env(inputs, batch_size)

        object_points = self.compute_swept_object_points(inputs)
        res = inputs['res']
        # get all components of the state as a set of points
        # in general this should be the swept volume, and should include the robot
        object_points_occupancy = lookup_points_in_vg(object_points, local_env, res, local_origin_point, batch_size)
        return inputs_aug, res, object_points, object_points_occupancy, local_env, local_origin_point

    def sample_object_transformations(self, batch_size):
        # sample a translation and rotation for the object state
        transformation_params = self.scenario.sample_object_augmentation_variables(1 * batch_size, self.seed)
        # pick the most valid transforms, via the learned object state augmentation validity model
        predicted_errors = self.invariance_model_wrapper.evaluate(transformation_params)
        _, best_transform_params_indices = tf.math.top_k(-predicted_errors, tf.cast(batch_size, tf.int32), sorted=False)
        best_transformation_params = tf.gather(transformation_params, best_transform_params_indices, axis=0)
        transformation_matrices = [transformations.compose_matrix(translate=p[:3], angles=p[3:]) for p in
                                   best_transformation_params]
        return tf.cast(transformation_matrices, tf.float32)

    def apply_object_augmentation_no_ik(self, transformation_matrices, inputs, batch_size, time):
        return self.scenario.apply_object_augmentation_no_ik(transformation_matrices,
                                                             inputs,
                                                             batch_size,
                                                             time,
                                                             self.local_env_helper.h,
                                                             self.local_env_helper.w,
                                                             self.local_env_helper.c)

    def apply_object_augmentation(self, transformation_matrices, inputs, batch_size, time):
        return self.scenario.apply_object_augmentation(transformation_matrices,
                                                       inputs,
                                                       batch_size,
                                                       time,
                                                       self.local_env_helper.h,
                                                       self.local_env_helper.w,
                                                       self.local_env_helper.c)

    def opt_object_augmentation(self,
                                inputs: Dict,
                                inputs_aug: Dict,
                                new_env: Dict,
                                obj_points,
                                object_points_occupancy,
                                res,
                                batch_size,
                                time):
        # viz new env
        if debug_aug():
            for b in debug_viz_batch_indices(self.batch_size):
                env_new_dict = {
                    'env': new_env['env'].numpy(),
                    'res': res[b].numpy(),
                }
                msg = environment_to_vg_msg(env_new_dict, frame='new_env_aug_vg', stamp=rospy.Time(0))
                self.debug.env_aug_pub1.publish(msg)

                send_voxelgrid_tf_origin_point_res(self.broadcaster,
                                                   origin_point=new_env['origin_point'],
                                                   res=res[b],
                                                   frame='new_env_aug_vg')
                # stepper.step()

        # Sample an initial random object transformation
        initial_transformation_params = self.scenario.sample_object_augmentation_variables(10 * batch_size, self.seed)
        # pick the most valid transforms, via the learned object state augmentation validity model
        predicted_errors = self.invariance_model_wrapper.evaluate(initial_transformation_params)
        _, best_transform_params_indices = tf.math.top_k(-predicted_errors, tf.cast(batch_size, tf.int32), sorted=False)
        initial_transformation_params = tf.gather(initial_transformation_params, best_transform_params_indices, axis=0)

        # optimization loop
        obj_transforms = tf.Variable(initial_transformation_params)  # [x,y,z,roll,pitch,yaw]
        variables = [obj_transforms]
        for i in range(self.max_steps):
            with tf.profiler.experimental.Trace('one_step_loop', i=i):
                with tf.GradientTape() as tape:
                    # obj_points is the set of points that define the object state, ie. the swept rope points
                    # to compute the object state constraints loss we need to transform this during each forward pass
                    # we also need to call apply_object_augmentation* at the end
                    # to update the rest of the "state" which is
                    # input to the network
                    transformation_matrices = transformation_params_to_matrices(obj_transforms, batch_size)
                    obj_points_aug = transform_points_3d(transformation_matrices[:, None], obj_points)

                    env_points_full = occupied_voxels_to_points(new_env['env'], new_env['res'], new_env['origin_point'])
                    env_points_sparse = subsample_points(env_points_full, self.env_subsample)

                    # compute repel and attract loss between the environment points and the obj_points_aug
                    attract_mask = object_points_occupancy  # assumed to already be either 0.0 or 1.0
                    dists = pairwise_squared_distances(env_points_sparse, obj_points_aug)
                    min_dist = tf.reduce_min(dists, axis=1)
                    min_dist_indices = tf.argmin(dists, axis=1)
                    nearest_env_points = tf.gather(env_points_sparse, min_dist_indices)

                    attract_loss = min_dist * self.attract_weight
                    repel_loss = self.barrier_func(min_dist) * self.repel_weight

                    attract_repel_loss_per_point = attract_mask * attract_loss + (1 - attract_mask) * repel_loss

                    invariance_loss = self.invariance_weight * self.invariance_model_wrapper.evaluate(obj_transforms)

                    robot_base_penetration_loss = self.robot_base_penetration_loss(obj_points_aug)

                    ground_penetration_loss = self.ground_penetration_loss(obj_points_aug)

                    losses = [
                        tf.reduce_mean(attract_repel_loss_per_point, axis=-1),
                        tf.reduce_mean(ground_penetration_loss, axis=-1),
                        tf.reduce_mean(robot_base_penetration_loss, axis=-1),
                        invariance_loss,
                    ]
                    loss = tf.reduce_mean(tf.add_n(losses))

                    # min_dists = MinDists(min_attract_dist_b, min_repel_dist_b, min_robot_repel_dist_b)
                    # env_opt_debug_vars = EnvOptDebugVars(nearest_attract_env_points, nearest_repel_points,
                    #                                      nearest_robot_repel_points)
                    # env_points_b = EnvPoints(env_points_b, env_points_b_sparse)
                gradients = tape.gradient(loss, variables)

            if debug_aug_sgd():
                stepper = RvizSimpleStepper()
                scale = 0.005
                for b in debug_viz_batch_indices(batch_size):
                    repel_indices = tf.squeeze(tf.where(1 - object_points_occupancy[b]), -1)
                    attract_indices = tf.squeeze(tf.where(object_points_occupancy[b]), -1)

                    attract_points = tf.gather(obj_points[b], attract_indices).numpy()
                    repel_points = tf.gather(obj_points[b], repel_indices).numpy()
                    attract_points_aug = tf.gather(obj_points_aug[b], attract_indices).numpy()
                    repel_points_aug = tf.gather(obj_points_aug[b], repel_indices).numpy()
                    nearest_attract_env_points = tf.gather(nearest_env_points[b], attract_indices).numpy()
                    nearest_repel_env_points = tf.gather(nearest_env_points[b], repel_indices).numpy()

                    self.scenario.plot_points_rviz(attract_points, label='attract', color='g', scale=scale)
                    self.scenario.plot_points_rviz(repel_points, label='repel', color='r', scale=scale)
                    self.scenario.plot_points_rviz(attract_points_aug, label='attract_aug', color='g', scale=scale)
                    self.scenario.plot_points_rviz(repel_points_aug, label='repel_aug', color='r', scale=scale)

                    self.scenario.plot_lines_rviz(nearest_attract_env_points, attract_points_aug,
                                                  label='attract_correspondence', color='g')
                    self.scenario.plot_lines_rviz(nearest_repel_env_points, repel_points_aug,
                                                  label='repel_correspondence', color='r')
                    # stepper.step()

            clipped_grads_and_vars = self.clip_env_aug_grad(gradients, variables)
            self.opt.apply_gradients(grads_and_vars=clipped_grads_and_vars)

            # check termination criteria
            squared_res_expanded = tf.square(res)[:, None]
            attract_satisfied = tf.cast(min_dist < squared_res_expanded, tf.float32)
            repel_satisfied = tf.cast(min_dist > squared_res_expanded, tf.float32)
            constraints_satisfied = attract_mask * attract_satisfied + (1 - attract_mask) * repel_satisfied
            constraints_satisfied = tf.reduce_all(tf.cast(constraints_satisfied, tf.bool), axis=-1)

            grad_norm = tf.linalg.norm(gradients[0], axis=-1)
            step_size_i = grad_norm * self.step_size
            can_terminate = self.can_terminate(constraints_satisfied, step_size_i)
            can_terminate = tf.reduce_all(can_terminate)
            if can_terminate:
                break

        # this updates other representations of state/action that are fed into the network
        _, object_aug_update, local_origin_point_aug, local_center_aug = self.apply_object_augmentation_no_ik(
            transformation_matrices,
            inputs,
            batch_size,
            time)
        inputs_aug.update(object_aug_update)

        if debug_aug_sgd():
            self.debug.send_position_transform(local_origin_point_aug[b], 'local_origin_point_aug')
            self.debug.send_position_transform(local_center_aug[b], 'local_center_aug')

        new_env_repeated = repeat(new_env, repetitions=batch_size, axis=0, new_axis=True)
        local_env_aug, _ = self.local_env_helper.get(local_center_aug, new_env_repeated, batch_size)
        # NOTE: after local optimization, enforce the constraint
        #  one way would be to force voxels with attract points are on and voxels with repel points are off
        #  another would be to "give up" and use the un-augmented datapoint

        local_env_aug_fixed = []
        local_env_aug_fix_deltas = []
        for b in range(batch_size):
            attract_indices = tf.squeeze(tf.where(object_points_occupancy[b]), axis=1)
            repel_indices = tf.squeeze(tf.where(1 - object_points_occupancy[b]), axis=1)
            attract_points_aug = tf.gather(obj_points_aug[b], attract_indices)
            repel_points_aug = tf.gather(obj_points_aug[b], repel_indices)
            attract_vg = self.points_to_voxel_grid_res_origin_point(attract_points_aug, res[b],
                                                                    local_origin_point_aug[b])
            repel_vg = self.points_to_voxel_grid_res_origin_point(repel_points_aug, res[b], local_origin_point_aug[b])
            # NOTE: the order of operators here is arbitrary, it gives different output, but I doubt it matters
            local_env_aug_fixed_b = subtract(binary_or(local_env_aug[b], attract_vg), repel_vg)
            local_env_aug_fix_delta = tf.reduce_sum(tf.abs(local_env_aug_fixed_b - local_env_aug[b]))
            local_env_aug_fix_deltas.append(local_env_aug_fix_delta)
            local_env_aug_fixed.append(local_env_aug_fixed_b)
        local_env_aug_fixed = tf.stack(local_env_aug_fixed, axis=0)
        local_env_aug_fix_deltas = tf.stack(local_env_aug_fix_deltas, axis=0)
        self.local_env_aug_fix_delta = possibly_none_concat(self.local_env_aug_fix_delta, local_env_aug_fix_deltas,
                                                            axis=0)

        return inputs_aug, local_origin_point_aug, local_center_aug, local_env_aug_fixed

    def opt_new_env_augmentation(self,
                                 inputs_aug: Dict,
                                 new_env: Dict,
                                 object_points_aug,
                                 robot_points_aug,
                                 object_points_occupancy,
                                 robot_points_occupancy,  # might be used in the future
                                 res,
                                 local_origin_point_aug,
                                 batch_size):
        """

        Args:
            new_env: [b, h, w, c]
            object_points_aug: [b, n, 3], in same frame as local_origin_point_aug (i.e. robot or world frame)
                    The set of points in the swept volume of the object, possibly augmented
            robot_points_aug: [b, n, 3], in same frame as local_origin_point_aug (i.e. robot or world frame)
                    The set of points in the swept volume of the robot, possibly augmented
            object_points_occupancy: [b, n]
            robot_points_occupancy: [b, n]
            res: [b]
            local_origin_point_aug: [b, 3]
            batch_size: int

        Returns: [b, h, w, c]

        """
        local_env_new_center = self.sample_local_env_position(new_env, batch_size)
        local_env_new, local_env_new_origin_point = self.local_env_helper.get(local_env_new_center, new_env, batch_size)
        # viz new env
        if debug_aug():
            for b in debug_viz_batch_indices(self.batch_size):
                self.debug.send_position_transform(local_env_new_center[b], 'local_env_new_center')

                send_voxelgrid_tf_origin_point_res(self.broadcaster,
                                                   origin_point=local_env_new_origin_point[b],
                                                   res=res[b],
                                                   frame='local_env_new_vg')

                bbox_msg = grid_to_bbox(rows=self.local_env_helper.h,
                                        cols=self.local_env_helper.w,
                                        channels=self.local_env_helper.c,
                                        resolution=res[b].numpy())
                bbox_msg.header.frame_id = 'local_env_new_vg'

                self.debug.local_env_new_bbox_pub.publish(bbox_msg)

                env_new_dict = {
                    'env': new_env['env'][b].numpy(),
                    'res': res[b].numpy(),
                }
                msg = environment_to_vg_msg(env_new_dict, frame='new_env_aug_vg', stamp=rospy.Time(0))
                self.debug.env_aug_pub1.publish(msg)

                send_voxelgrid_tf_origin_point_res(self.broadcaster,
                                                   origin_point=new_env['origin_point'][b],
                                                   res=res[b],
                                                   frame='new_env_aug_vg')

                # Show sample new local environment, in the frame of the original local env, the one we're augmenting
                local_env_new_dict = {
                    'env': local_env_new[b].numpy(),
                    'res': res[b].numpy(),
                }
                msg = environment_to_vg_msg(local_env_new_dict, frame='local_env_aug_vg', stamp=rospy.Time(0))
                self.debug.env_aug_pub2.publish(msg)

                send_voxelgrid_tf_origin_point_res(self.broadcaster,
                                                   origin_point=local_origin_point_aug[b],
                                                   res=res[b],
                                                   frame='local_env_aug_vg')

                # stepper.step()

        if debug_aug_sgd():
            stepper = RvizSimpleStepper()

        local_env_aug = []
        env_aug_valid = []

        translation_b = tf.Variable(tf.zeros(3, dtype=tf.float32), dtype=tf.float32)
        for b in range(batch_size):
            with tf.profiler.experimental.Trace('one_batch_loop', b=b):
                r_b = res[b]
                o_b = local_origin_point_aug[b]
                object_points_b = object_points_aug[b]
                robot_points_b = robot_points_aug[b]
                # NOTE: sub-sample because to speed up and avoid OOM.
                #  Unfortunately this also makes our no-robot-inside-env constraint approximate
                robot_points_b_sparse = subsample_points(robot_points_b, self.robot_subsample)
                object_occupancy_b = object_points_occupancy[b]
                env_points_b_initial_full = occupied_voxels_to_points(local_env_new[b], r_b, o_b)
                env_points_b_initial_sparse = subsample_points(env_points_b_initial_full, self.env_subsample)
                env_points_b_initial = EnvPoints(env_points_b_initial_full, env_points_b_initial_sparse)
                env_points_b = env_points_b_initial

                initial_is_attract_indices = tf.squeeze(tf.where(object_occupancy_b > 0.5), 1)
                initial_attract_points_b = tf.gather(object_points_b, initial_is_attract_indices)
                if tf.size(initial_is_attract_indices) == 0:
                    initial_translation_b = tf.zeros(3)
                else:
                    env_points_b_initial_mean = tf.reduce_mean(env_points_b_initial_full, axis=0)
                    initial_attract_points_b_mean = tf.reduce_mean(initial_attract_points_b, axis=0)
                    initial_translation_b = initial_attract_points_b_mean - env_points_b_initial_mean

                translation_b.assign(initial_translation_b)
                variables = [translation_b]

                hard_constraints_satisfied_b = False

                is_attract_indices = tf.squeeze(tf.where(object_occupancy_b > 0.5), 1)
                attract_points_b = tf.gather(object_points_b, is_attract_indices)

                is_repel_indices = tf.squeeze(tf.where(object_occupancy_b < 0.5), 1)
                repel_points_b = tf.gather(object_points_b, is_repel_indices)
                for i in range(self.max_steps):
                    with tf.profiler.experimental.Trace('one_step_loop', b=b, i=i):
                        if tf.size(env_points_b_initial_full) == 0:
                            hard_constraints_satisfied_b = True
                            break

                        with tf.GradientTape() as tape:
                            loss, min_dists, dbg, env_points_b = self.env_opt_forward(env_points_b_initial,
                                                                                      translation_b,
                                                                                      attract_points_b,
                                                                                      repel_points_b,
                                                                                      robot_points_b_sparse)

                        gradients = tape.gradient(loss, variables)

                        clipped_grads_and_vars = self.clip_env_aug_grad(gradients, variables)
                        self.opt.apply_gradients(grads_and_vars=clipped_grads_and_vars)

                        if debug_aug_sgd():
                            repel_close_indices = tf.squeeze(tf.where(min_dists.repel < self.barrier_upper_lim),
                                                             axis=-1)
                            robot_repel_close_indices = tf.squeeze(
                                tf.where(min_dists.robot_repel < self.barrier_upper_lim),
                                axis=-1)
                            nearest_repel_points_where_close = tf.gather(dbg.nearest_repel_points, repel_close_indices)
                            nearest_robot_repel_points_where_close = tf.gather(dbg.nearest_robot_repel_points,
                                                                               robot_repel_close_indices)
                            env_points_b_where_close = tf.gather(env_points_b.sparse, repel_close_indices)
                            env_points_b_where_close_to_robot = tf.gather(env_points_b.sparse,
                                                                          robot_repel_close_indices)
                            if b in debug_viz_batch_indices(batch_size):
                                t_for_robot_viz = 0
                                state_b_i = {
                                    'joint_names':     inputs_aug['joint_names'][b, t_for_robot_viz],
                                    'joint_positions': inputs_aug[add_predicted('joint_positions')][b, t_for_robot_viz],
                                }
                                self.scenario.plot_state_rviz(state_b_i, label='aug_opt')
                                self.scenario.plot_points_rviz(env_points_b.sparse, label='icp', color='grey',
                                                               scale=0.005)
                                self.scenario.plot_lines_rviz(dbg.nearest_attract_env_points, attract_points_b,
                                                              label='attract_correspondence', color='g')
                                self.scenario.plot_lines_rviz(nearest_repel_points_where_close,
                                                              env_points_b_where_close,
                                                              label='repel_correspondence', color='r')
                                self.scenario.plot_lines_rviz(nearest_robot_repel_points_where_close,
                                                              env_points_b_where_close_to_robot,
                                                              label='robot_repel_correspondence', color='orange')
                                # stepper.step()

                        squared_res = tf.square(res[b])
                        hard_repel_constraint_satisfied_b = tf.reduce_min(min_dists.repel) > squared_res
                        hard_robot_repel_constraint_satisfied_b = tf.reduce_min(min_dists.robot_repel) > squared_res
                        hard_attract_constraint_satisfied_b = tf.reduce_max(min_dists.attract) < squared_res

                        hard_constraints_satisfied_b = tf.reduce_all([hard_repel_constraint_satisfied_b,
                                                                      hard_robot_repel_constraint_satisfied_b,
                                                                      hard_attract_constraint_satisfied_b])
                        grad_norm = tf.linalg.norm(gradients)
                        step_size_b_i = grad_norm * self.step_size
                        if debug_aug_sgd():
                            if b in debug_viz_batch_indices(batch_size):
                                print(step_size_b_i, self.step_size_threshold, hard_constraints_satisfied_b)

                        can_terminate = self.can_terminate(hard_constraints_satisfied_b, step_size_b_i)
                        if can_terminate.numpy():
                            break

                local_env_aug_b = self.points_to_voxel_grid_res_origin_point(env_points_b.full, r_b, o_b)

                # NOTE: after local optimization, enforce the constraint
                #  one way would be to force voxels with attract points are on and voxels with repel points are off
                #  another would be to "give up" and use the un-augmented datapoint
                attract_vg_b = self.points_to_voxel_grid_res_origin_point(attract_points_b, r_b, o_b)
                repel_vg_b = self.points_to_voxel_grid_res_origin_point(repel_points_b, r_b, o_b)
                # NOTE: the order of operators here is arbitrary, it gives different output, but I doubt it matters
                local_env_aug_b = subtract(binary_or(local_env_aug_b, attract_vg_b), repel_vg_b)

                local_env_aug.append(local_env_aug_b)
                env_aug_valid.append(hard_constraints_satisfied_b)

        local_env_aug = tf.stack(local_env_aug)
        env_aug_valid = tf.cast(tf.stack(env_aug_valid), tf.float32)

        return env_aug_valid, local_env_aug

    def env_opt_forward(self,
                        env_points_b_initial: EnvPoints,
                        translation_b,
                        attract_points_b,
                        repel_points_b,
                        robot_points_b_sparse):
        env_points_b_sparse = env_points_b_initial.sparse + translation_b  # this expression must be inside the tape
        env_points_b = env_points_b_initial.full + translation_b

        # NOTE: these are SQUARED distances!
        attract_dists_b = pairwise_squared_distances(env_points_b_sparse, attract_points_b)
        min_attract_dist_indices_b = tf.argmin(attract_dists_b, axis=0, name='attract_argmin')
        min_attract_dist_b = tf.reduce_min(attract_dists_b, axis=0)
        nearest_attract_env_points = tf.gather(env_points_b_sparse, min_attract_dist_indices_b)
        attract_loss = reduce_mean_no_nan(min_attract_dist_b)

        repel_dists_b = pairwise_squared_distances(env_points_b_sparse, repel_points_b)
        min_repel_dist_indices_b = tf.argmin(repel_dists_b, axis=1, name='repel_argmin')
        min_repel_dist_b = tf.reduce_min(repel_dists_b, axis=1)
        nearest_repel_points = tf.gather(repel_points_b, min_repel_dist_indices_b)
        repel_loss = reduce_mean_no_nan(self.barrier_func(min_repel_dist_b))

        robot_repel_dists_b = pairwise_squared_distances(env_points_b_sparse, robot_points_b_sparse)
        min_robot_repel_dist_b = tf.reduce_min(robot_repel_dists_b, axis=1)
        min_robot_repel_dist_indices_b = tf.argmin(robot_repel_dists_b, axis=1, name='robot_repel_argmin')
        nearest_robot_repel_points = tf.gather(robot_points_b_sparse, min_robot_repel_dist_indices_b)
        robot_repel_loss = tf.reduce_mean(self.barrier_func(min_robot_repel_dist_b))

        loss = attract_loss * self.attract_weight + repel_loss + robot_repel_loss

        min_dists = MinDists(min_attract_dist_b, min_repel_dist_b, min_robot_repel_dist_b)
        env_opt_debug_vars = EnvOptDebugVars(nearest_attract_env_points, nearest_repel_points,
                                             nearest_robot_repel_points)
        env_points_b = EnvPoints(env_points_b, env_points_b_sparse)
        return (
            loss,
            min_dists,
            env_opt_debug_vars,
            env_points_b,
        )

    def can_terminate(self, constraints_satisfied, step_size):
        can_terminate = tf.logical_or(step_size < self.step_size_threshold, constraints_satisfied)
        return can_terminate

    def clip_env_aug_grad(self, gradients, variables):
        def _clip(g):
            # we want grad_clip to be as close to in meters as possible, so here we scale by step size
            c = self.grad_clip / self.step_size
            return tf.clip_by_value(g, -c, c)

        return [(_clip(g), v) for (g, v) in zip(gradients, variables)]

    def barrier_func(self, min_dists_b):
        z = tf.math.log(self.barrier_scale * min_dists_b + self.barrier_epsilon)
        # of course this additive term doesn't affect the gradient, but it makes hyper-parameters more interpretable
        return tf.maximum(-z, -self.log_cutoff) + self.log_cutoff

    def points_to_voxel_grid_res_origin_point(self, points, res, origin_point):
        return points_to_voxel_grid_res_origin_point(points,
                                                     res,
                                                     origin_point,
                                                     self.local_env_helper.h,
                                                     self.local_env_helper.w,
                                                     self.local_env_helper.c)

    def get_new_env(self, example):
        if add_new('env') not in example:
            example[add_new('env')] = example['env']
            example[add_new('extent')] = example['extent']
            example[add_new('origin_point')] = example['origin_point']
            example[add_new('res')] = example['res']
            example[add_new('scene_msg')] = example['scene_msg']
        new_env = {
            'env':          example[add_new('env')],
            'extent':       example[add_new('extent')],
            'origin_point': example[add_new('origin_point')],
            'res':          example[add_new('res')],
            'scene_msg':    example[add_new('scene_msg')],
        }
        return new_env

    def sample_local_env_position(self, example, batch_size):
        # NOTE: for my specific implementation of state_to_local_env_pose,
        #  sampling random states and calling state_to_local_env_pose is equivalent to sampling a point in the extent
        extent = tf.reshape(example['extent'], [batch_size, 3, 2])
        extent_lower = tf.gather(extent, 0, axis=-1)
        extent_upper = tf.gather(extent, 1, axis=-1)
        local_env_center = self.gen.uniform([batch_size, 3], extent_lower, extent_upper)

        return local_env_center

    def do_augmentation(self):
        return self.hparams is not None

    def compute_swept_robot_points(self, inputs, batch_size):
        robot_points_0 = self.vg_info.make_robot_points_batched(batch_size, inputs, 0)
        robot_points_1 = self.vg_info.make_robot_points_batched(batch_size, inputs, 1)
        robot_points = tf.linspace(robot_points_0, robot_points_1, self.num_robot_interp, axis=1)
        robot_points = tf.reshape(robot_points, [batch_size, -1, 3])
        return robot_points

    def compute_swept_object_points(self, inputs):
        points_state_keys = [add_predicted(k) for k in self.points_state_keys]
        batch_size = inputs['batch_size']

        def _make_points(k, t):
            v = inputs[k][:, t]
            points = tf.reshape(v, [batch_size, -1, 3])
            points = densify_points(batch_size, points)
            return points

        object_points_0 = {k: _make_points(k, 0) for k in points_state_keys}
        object_points_1 = {k: _make_points(k, 1) for k in points_state_keys}

        def _linspace(k):
            return tf.linspace(object_points_0[k], object_points_1[k], self.num_object_interp, axis=1)

        swept_object_points = tf.concat([_linspace(k) for k in points_state_keys], axis=2)
        swept_object_points = tf.reshape(swept_object_points, [batch_size, -1, 3])

        return swept_object_points

    def get_local_env(self, input_dict, batch_size):
        state_0 = {k: input_dict[add_predicted(k)][:, 0] for k in self.state_keys}

        # NOTE: to be more general, this should return a pose not just a point/position
        local_env_center = self.scenario.local_environment_center_differentiable(state_0)
        environment = {k: input_dict[k] for k in ['env', 'origin_point', 'res', 'extent']}
        local_env, local_origin_point = self.local_env_helper.get(local_env_center, environment, batch_size)

        return local_env, local_origin_point

    def solve_ik(self, inputs: Dict, inputs_aug: Dict, new_env: Dict, batch_size: int):
        left_gripper_points_aug = inputs_aug[add_predicted('left_gripper')]
        right_gripper_points_aug = inputs_aug[add_predicted('right_gripper')]
        deserialize_scene_msg(new_env)

        # run ik, try to find collision free solution
        joint_names = to_list_of_strings(inputs['joint_names'][0, 0].numpy().tolist())
        scene_msg = new_env['scene_msg']
        default_robot_positions = inputs[add_predicted('joint_positions')][:, 0]
        joint_positions_aug_, is_ik_valid = self.ik_solver.solve(scene_msg=scene_msg,
                                                                 joint_names=joint_names,
                                                                 default_robot_positions=default_robot_positions,
                                                                 left_target_position=left_gripper_points_aug[:, 0],
                                                                 right_target_position=right_gripper_points_aug[:, 0],
                                                                 batch_size=batch_size)
        joint_positions_aug_start = joint_positions_aug_

        # then run the jacobian follower to compute the second new position
        tool_names = [self.scenario.robot.left_tool_name, self.scenario.robot.right_tool_name]
        preferred_tool_orientations = self.scenario.get_preferred_tool_orientations(tool_names)
        joint_positions_aug = []
        reached = []
        for b in range(batch_size):
            scene_msg_b = scene_msg[b]
            grippers_end_b = [left_gripper_points_aug[b, 1][None].numpy(), right_gripper_points_aug[b, 1][None].numpy()]
            joint_positions_aug_start_b = joint_positions_aug_start[b]
            start_joint_state_b = JointState(name=joint_names, position=joint_positions_aug_start_b.numpy())
            empty_scene_msg_b, start_robot_state_b = merge_joint_state_and_scene_msg(scene_msg_b,
                                                                                     start_joint_state_b)
            plan_to_end, reached_end_b = self.scenario.robot.jacobian_follower.plan(
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

        if debug_ik():
            print(f"valid % = {tf.reduce_mean(is_ik_valid)}")
        return joint_positions_aug, is_ik_valid

    def ground_penetration_loss(self, obj_points_aug):
        obj_points_aug_z = obj_points_aug[:, :, 2]
        ground_z = -0.415  # FIXME: hardcoded, copied from the gazebo world file
        return self.ground_penetration_weight * tf.maximum(0, ground_z - obj_points_aug_z)

    def robot_base_penetration_loss(self, obj_points_aug):
        obj_points_aug_y = obj_points_aug[:, :, 1]
        base_y = 0.15  # FIXME: hardcoded
        return self.robot_base_penetration_weight * tf.maximum(0, base_y - obj_points_aug_y)
