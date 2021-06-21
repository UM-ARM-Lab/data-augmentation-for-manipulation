import pathlib
from typing import Dict, List

import tensorflow as tf
import tensorflow_probability as tfp
import transformations

import rospy
from learn_invariance.invariance_model_wrapper import InvarianceModelWrapper
from link_bot_classifiers.classifier_debugging import ClassifierDebugging
from link_bot_classifiers.local_env_helper import LocalEnvHelper
from link_bot_classifiers.make_voxelgrid_inputs import VoxelgridInfo
from link_bot_data.dataset_utils import add_new, add_predicted
from link_bot_pycommon.bbox_visualization import grid_to_bbox
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from link_bot_pycommon.grid_utils import lookup_points_in_vg, send_voxelgrid_tf_origin_point_res, environment_to_vg_msg, \
    occupied_voxels_to_points, subtract, binary_or
from link_bot_pycommon.pycommon import densify_points
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from merrrt_visualization.rviz_animation_controller import RvizSimpleStepper
from moonshine.geometry import transform_points_3d, pairwise_squared_distances
from moonshine.optimization import log_barrier
from moonshine.raster_3d import points_to_voxel_grid_res_origin_point

DEBUG_AUG = False
DEBUG_AUG_SGD = False


def subsample_points(points, fraction):
    """

    Args:
        points: [n, 3]
        fraction: from 0.0 to 1.0

    Returns:

    """
    n_take_every = int(1 / fraction)
    return points[::n_take_every]


class AugmentationOptimization:

    def __init__(self,
                 scenario: ScenarioWithVisualization,
                 debug: ClassifierDebugging,
                 local_env_helper: LocalEnvHelper,
                 vg_info: VoxelgridInfo,
                 points_state_keys: List[str],
                 hparams: Dict,
                 batch_size: int,
                 action_keys: List[str]):
        self.action_keys = action_keys
        self.hparams = hparams.get('augmentation', None)
        self.points_state_keys = points_state_keys
        self.batch_size = batch_size
        self.scenario = scenario
        self.vg_info = vg_info
        self.debug = debug
        self.local_env_helper = local_env_helper
        self.broadcaster = self.scenario.tf.tf_broadcaster

        self.max_steps = 18
        self.num_interp = 5
        self.gen = tf.random.Generator.from_seed(0)
        self.seed = tfp.util.SeedStream(1, salt="nn_classifier_aug")
        self.step_size = 10.0
        self.opt = tf.keras.optimizers.SGD(self.step_size)
        self.step_size_threshold = 0.002  # stopping criteria, how far the env moved (meters)
        self.barrier_upper_lim = tf.square(0.06)  # stops repelling points from pushing after this distance
        self.barrier_scale = 0.05  # scales the gradients for the repelling points
        self.grad_clip = 0.25  # max dist step the env aug update can take

        if self.hparams is not None:
            invariance_model_path = pathlib.Path(self.hparams['invariance_model'])
            self.invariance_model_wrapper = InvarianceModelWrapper(invariance_model_path, self.batch_size,
                                                                   self.scenario)

    def augmentation_optimization(self,
                                  inputs: Dict,
                                  batch_size,
                                  time):
        # TODO: rewrite this so that we do a out-of-place update, i.e returning copied dicts from the "apply" and "opt" functions, and then check both object_aug_valid and env_aug_valid together
        #  visualize everything to make sure that when either is invalid, we "abort" the augmentation, and the return dict (which is actually what's used) is correct
        #  also check the "animate voxel grids" that the state components of the voxel_grids is correct. I think right now it isn't?
        inputs_aug = {
            'res':                  inputs['res'],
            'extent':               inputs['extent'],
            'origin_point':         inputs['origin_point'],
            'env':                  inputs['env'],
            'is_close':             inputs['is_close'],
            'batch_size':           inputs['batch_size'],
            'time':                 inputs['time'],
            add_predicted('stdev'): inputs[add_predicted('stdev')],
        }

        object_points = inputs['swept_object_points']
        res = inputs['res']
        local_origin_point = inputs['local_origin_point']
        local_env = inputs['voxel_grids'][:, 0, :, :, :, 0]  # just use 0 because it's the same at all time steps
        # get all components of the state as a set of points
        # in general this should be the swept volume, and should include the robot
        object_points_occupancy = lookup_points_in_vg(object_points, local_env, res, local_origin_point, batch_size)

        transformation_matrices = self.sample_object_transformations(batch_size)
        object_aug_valid, object_aug_update = self.scenario.apply_object_augmentation(transformation_matrices,
                                                                                      inputs,
                                                                                      batch_size,
                                                                                      time,
                                                                                      self.local_env_helper.h,
                                                                                      self.local_env_helper.w,
                                                                                      self.local_env_helper.c)
        inputs_aug.update(object_aug_update)
        local_origin_point_aug = inputs_aug['local_origin_point']

        # this was just updated by apply_state_augmentation
        if DEBUG_AUG:
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

        object_points_aug = transform_points_3d(transformation_matrices[:, None], object_points)
        robot_points_aug = self.compute_swept_robot_points(inputs_aug, batch_size)

        if DEBUG_AUG:
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

        # robot_points_occupancy = lookup_points_in_vg(robot_points_aug, local_env, res, local_origin_point_aug, batch_size)
        new_env = self.get_new_env(inputs)
        env_aug_valid, local_env_aug = self.opt_new_env_augmentation(inputs_aug,
                                                                     new_env,
                                                                     object_points_aug,
                                                                     robot_points_aug,
                                                                     object_points_occupancy,
                                                                     None,  # robot_points_occupancy,
                                                                     res,
                                                                     local_origin_point_aug,
                                                                     batch_size)
        # NOTE: now we to re-compute the voxel_grids, since the object state, robot state, and local_env have changed
        voxel_grids_aug = self.vg_info.make_voxelgrid_inputs(inputs_aug, local_env_aug, local_origin_point_aug,
                                                             batch_size, time)
        inputs_aug['voxel_grids'] = voxel_grids_aug

        if DEBUG_AUG:
            stepper = RvizSimpleStepper()
            for b in debug_viz_batch_indices(batch_size):
                _aug_dict = {
                    'env':          inputs_aug['voxel_grids'][b, 0, :, :, :, 0].numpy(),
                    'origin_point': local_origin_point_aug[b].numpy(),
                    'res':          res[b].numpy(),
                }
                msg = environment_to_vg_msg(_aug_dict, frame='local_env_aug_vg', stamp=rospy.Time(0))
                self.debug.env_aug_pub5.publish(msg)
                send_voxelgrid_tf_origin_point_res(self.broadcaster,
                                                   local_origin_point_aug[b],
                                                   res[b],
                                                   frame='local_env_aug_vg')

                self.debug.plot_state_action_rviz(inputs_aug, b, 'aug', color='blue')
                # stepper.step()  # FINAL AUG (not necessarily what the network sees, only if valid)

        is_valid = env_aug_valid * object_aug_valid
        # FIXME: this is so hacky
        keys_aug = ['voxel_grids', 'local_origin_point', add_predicted('joint_positions')]
        keys_aug += self.action_keys
        keys_aug += [add_predicted(psk) for psk in self.points_state_keys]
        for k in keys_aug:
            v = inputs_aug[k]
            is_valid_expanded = tf.reshape(is_valid, [batch_size] + [1] * (v.ndim - 1))
            inputs_aug[k] = is_valid_expanded * inputs_aug[k] + (1 - is_valid_expanded) * inputs[k]
        return inputs_aug

    def sample_object_transformations(self, batch_size):
        # sample a translation and rotation for the object state
        transformation_params = self.scenario.sample_object_augmentation_variables(10 * batch_size, self.seed)
        # pick the most valid transforms, via the learned object state augmentation validity model
        predicted_errors = self.invariance_model_wrapper.evaluate(transformation_params)
        _, best_transform_params_indices = tf.math.top_k(-predicted_errors, tf.cast(batch_size, tf.int32), sorted=False)
        best_transformation_params = tf.gather(transformation_params, best_transform_params_indices, axis=0)
        transformation_matrices = [transformations.compose_matrix(translate=p[:3], angles=p[3:]) for p in
                                   best_transformation_params]
        return tf.cast(transformation_matrices, tf.float32)

    def opt_new_env_augmentation(self,
                                 inputs_aug: Dict,
                                 new_env: Dict,
                                 object_points_aug,
                                 robot_points_aug,
                                 object_points_occupancy,
                                 robot_points_occupancy,
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
        if DEBUG_AUG:
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

        if DEBUG_AUG_SGD:
            stepper = RvizSimpleStepper()

        nearest_attract_env_points = None
        nearest_repel_points = None
        attract_points_b = None
        repel_points_b = None
        local_env_aug = []
        env_aug_valid = []

        for b in range(batch_size):
            r_b = res[b]
            o_b = local_origin_point_aug[b]
            object_points_b = object_points_aug[b]
            robot_points_b = robot_points_aug[b]
            robot_points_b = subsample_points(robot_points_b, 0.5)  # sub-sample because to speed up and avoid OOM
            object_occupancy_b = object_points_occupancy[b]
            env_points_b_initial = occupied_voxels_to_points(local_env_new[b], r_b, o_b)
            env_points_b = env_points_b_initial

            initial_is_attract_indices = tf.squeeze(tf.where(object_occupancy_b > 0.5), 1)
            initial_attract_points_b = tf.gather(object_points_b, initial_is_attract_indices)
            if tf.size(initial_is_attract_indices) == 0:
                initial_translation_b = tf.zeros(3)
            else:
                env_points_b_initial_mean = tf.reduce_mean(env_points_b_initial, axis=0)
                initial_attract_points_b_mean = tf.reduce_mean(initial_attract_points_b, axis=0)
                initial_translation_b = initial_attract_points_b_mean - env_points_b_initial_mean
            translation_b = tf.Variable(initial_translation_b, dtype=tf.float32)
            variables = [translation_b]
            hard_constraints_satisfied_b = 0
            for i in range(self.max_steps):
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(translation_b)
                    env_points_b = env_points_b_initial + translation_b
                    env_points_b_sparse = subsample_points(env_points_b, 0.5)

                    is_attract_indices = tf.squeeze(tf.where(object_occupancy_b > 0.5), 1)
                    attract_points_b = tf.gather(object_points_b, is_attract_indices)

                    if tf.logical_or(tf.size(is_attract_indices) == 0, tf.size(env_points_b) == 0):
                        attract_loss = 0
                        min_attract_dist_b = 0.0
                    else:
                        # NOTE: these are SQUARED distances!
                        attract_dists_b = pairwise_squared_distances(env_points_b_sparse, attract_points_b)
                        min_attract_dist_indices_b = tf.argmin(attract_dists_b, axis=0, name='attract_argmin')
                        min_attract_dist_b = tf.reduce_min(attract_dists_b, axis=0)
                        nearest_attract_env_points = tf.gather(env_points_b_sparse, min_attract_dist_indices_b)
                        attract_loss = tf.reduce_mean(min_attract_dist_b)

                    is_repel_indices = tf.squeeze(tf.where(object_occupancy_b < 0.5), 1)
                    repel_points_b = tf.gather(object_points_b, is_repel_indices)

                    if tf.size(is_repel_indices) == 0:
                        repel_loss = 0
                        min_repel_dist_b = 0.0
                    else:
                        repel_dists_b = pairwise_squared_distances(env_points_b_sparse, repel_points_b)
                        min_repel_dist_indices_b = tf.argmin(repel_dists_b, axis=1, name='repel_argmin')
                        min_repel_dist_b = tf.reduce_min(repel_dists_b, axis=1)
                        nearest_repel_points = tf.gather(repel_points_b, min_repel_dist_indices_b)
                        repel_loss = tf.reduce_mean(self.barrier_func(min_repel_dist_b))

                    robot_repel_dists_b = pairwise_squared_distances(env_points_b_sparse, robot_points_b)
                    min_robot_repel_dist_b = tf.reduce_min(robot_repel_dists_b, axis=1)
                    min_robot_repel_dist_indices_b = tf.argmin(robot_repel_dists_b, axis=1, name='robot_repel_argmin')
                    nearest_robot_repel_points = tf.gather(robot_points_b, min_robot_repel_dist_indices_b)
                    robot_repel_loss = tf.reduce_mean(self.barrier_func(min_robot_repel_dist_b))

                    loss = attract_loss * 0.05 + repel_loss + robot_repel_loss

                if DEBUG_AUG_SGD:
                    repel_close_indices = tf.squeeze(tf.where(min_repel_dist_b < self.barrier_upper_lim), axis=-1)
                    robot_repel_close_indices = tf.squeeze(tf.where(min_robot_repel_dist_b < self.barrier_upper_lim),
                                                           axis=-1)
                    nearest_repel_points_where_close = tf.gather(nearest_repel_points, repel_close_indices)
                    nearest_robot_repel_points_where_close = tf.gather(nearest_robot_repel_points,
                                                                       robot_repel_close_indices)
                    env_points_b_where_close = tf.gather(env_points_b_sparse, repel_close_indices)
                    env_points_b_where_close_to_robot = tf.gather(env_points_b_sparse, robot_repel_close_indices)
                    if b in debug_viz_batch_indices(batch_size):
                        t_for_robot_viz = 0
                        state_b_i = {
                            'joint_names':     inputs_aug['joint_names'][b, t_for_robot_viz],
                            'joint_positions': inputs_aug[add_predicted('joint_positions')][b, t_for_robot_viz],
                        }
                        self.scenario.plot_state_rviz(state_b_i, label='aug_opt')
                        self.scenario.plot_points_rviz(env_points_b_sparse, label='icp', color='grey', scale=0.005)
                        self.scenario.plot_lines_rviz(nearest_attract_env_points, attract_points_b,
                                                      label='attract_correspondence', color='g')
                        self.scenario.plot_lines_rviz(nearest_repel_points_where_close,
                                                      env_points_b_where_close,
                                                      label='repel_correspondence', color='r')
                        self.scenario.plot_lines_rviz(nearest_robot_repel_points_where_close,
                                                      env_points_b_where_close_to_robot,
                                                      label='robot_repel_correspondence', color='orange')
                        # stepper.step()

                gradients = tape.gradient(loss, variables)

                clipped_grads_and_vars = self.clip_env_aug_grad(gradients, variables)
                self.opt.apply_gradients(grads_and_vars=clipped_grads_and_vars)

                hard_repel_constraint_satisfied_b = tf.reduce_min(min_repel_dist_b) > tf.square(res[b])
                hard_robot_repel_constraint_satisfied_b = tf.reduce_min(min_robot_repel_dist_b) > tf.square(res[b])
                hard_attract_constraint_satisfied_b = tf.reduce_max(min_attract_dist_b) < tf.square(res[b])

                hard_constraints_satisfied_b = tf.reduce_all([hard_repel_constraint_satisfied_b,
                                                              hard_robot_repel_constraint_satisfied_b,
                                                              hard_attract_constraint_satisfied_b])
                grad_norm = tf.linalg.norm(gradients)
                step_size_b_i = grad_norm * self.step_size
                if DEBUG_AUG_SGD:
                    if b in debug_viz_batch_indices(batch_size):
                        print(step_size_b_i, self.step_size_threshold, hard_constraints_satisfied_b)

                if tf.logical_or(step_size_b_i < self.step_size_threshold, hard_constraints_satisfied_b):
                    break
            local_env_aug_b = self.points_to_voxel_grid_res_origin_point(env_points_b, r_b, o_b)

            # NOTE: # after local optimization, enforce the constraint
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

    def can_terminate(self, step_size_b_i, hard_constraints_satisfied_b):
        n = tf.logical_or(step_size_b_i < self.step_size_threshold, hard_constraints_satisfied_b).numpy()
        return bool(n)

    def clip_env_aug_grad(self, gradients, variables):
        def _clip(g):
            # we want grad_clip to be as close to in meters as possible, so here we scale by step size
            c = self.grad_clip / self.step_size
            return tf.clip_by_value(g, -c, c)

        return [(_clip(g), v) for (g, v) in zip(gradients, variables)]

    def barrier_func(self, min_dists_b):
        return log_barrier(min_dists_b, scale=self.barrier_scale, cutoff=self.barrier_upper_lim)

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
        new_env = {
            'env':          example[add_new('env')],
            'extent':       example[add_new('extent')],
            'origin_point': example[add_new('origin_point')],
            'res':          example[add_new('res')],
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
        robot_points = tf.linspace(robot_points_0, robot_points_1, self.num_interp, axis=1)
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
            return tf.linspace(object_points_0[k], object_points_1[k], self.num_interp, axis=1)

        swept_object_points = tf.concat([_linspace(k) for k in points_state_keys], axis=2)
        swept_object_points = tf.reshape(swept_object_points, [batch_size, -1, 3])

        return swept_object_points
