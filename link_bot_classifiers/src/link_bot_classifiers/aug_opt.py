import pathlib
from typing import Dict, List

import tensorflow as tf
import tensorflow_probability as tfp
import transformations
from pyjacobian_follower import IkParams

import rospy
from arm_robots.robot_utils import merge_joint_state_and_scene_msg
from learn_invariance.invariance_model_wrapper import InvarianceModelWrapper
from link_bot_classifiers.aug_opt_ik import AugOptIk
from link_bot_classifiers.aug_opt_manual import opt_object_manual
from link_bot_classifiers.aug_opt_utils import debug_aug, debug_ik, pick_best_params
from link_bot_classifiers.aug_opt_v3 import opt_object_augmentation3
from link_bot_classifiers.aug_opt_v5 import opt_object_augmentation5
from link_bot_classifiers.aug_opt_v6 import opt_object_augmentation6
from link_bot_classifiers.classifier_debugging import ClassifierDebugging
from link_bot_classifiers.local_env_helper import LocalEnvHelper
from link_bot_classifiers.make_voxelgrid_inputs import VoxelgridInfo
from link_bot_data.dataset_utils import add_new, add_predicted, deserialize_scene_msg
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from link_bot_pycommon.grid_utils import lookup_points_in_vg, send_voxelgrid_tf_origin_point_res, environment_to_vg_msg
from link_bot_pycommon.pycommon import densify_points
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.moonshine_utils import to_list_of_strings, possibly_none_concat
from moonshine.raster_3d import points_to_voxel_grid_res_origin_point
from sensor_msgs.msg import JointState


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

        self.robot_subsample = 0.5
        self.env_subsample = 0.25
        self.num_object_interp = 2  # must be >=2
        self.num_robot_interp = 2  # must be >=2
        self.seed_int = 4 if self.hparams is None or 'seed' not in self.hparams else self.hparams['seed']
        self.gen = tf.random.Generator.from_seed(self.seed_int)
        self.seed = tfp.util.SeedStream(self.seed_int + 1, salt="nn_classifier_aug")
        self.step_size_threshold = 0.0003  # stopping criteria, how far the env moved (meters)
        self.barrier_cut_off = 0.06  # stop repelling loss after this (squared) distance (meters)
        self.barrier_epsilon = 0.01
        self.grad_clip = 0.25  # max dist step the env aug update can take
        self.ground_penetration_weight = 1.0
        self.robot_base_penetration_weight = 1.0
        self.ik_solver = None

        if self.do_augmentation():
            ik_params = IkParams(rng_dist=self.hparams.get("rand_dist", 0.1),
                                 max_collision_check_attempts=self.hparams.get("max_collision_check_attempts", 1))
            self.ik_solver = AugOptIk(scenario.robot, ik_params=ik_params)

            self.aug_type = self.hparams['type']

            # Precompute this for speed
            self.barrier_epsilon = 0.01

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
        return self.augmentation_optimization2(inputs, batch_size, time)

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
        if self.aug_type in ['optimization2', 'v3']:
            aug_f = opt_object_augmentation3
        elif self.aug_type in ['v5']:
            aug_f = opt_object_augmentation5
        elif self.aug_type in ['v6']:
            aug_f = opt_object_augmentation6
        elif self.aug_type in ['manual']:
            aug_f = opt_object_manual
        else:
            raise NotImplementedError()
        inputs_aug, local_origin_point_aug, local_center_aug, local_env_aug, is_obj_aug_valid = \
            aug_f(self,
                  inputs,
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

        is_valid = is_ik_valid * is_obj_aug_valid
        self.is_valids = possibly_none_concat(self.is_valids, is_valid, axis=0)

        if debug_aug():
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

                self.debug.plot_state_rviz(inputs_aug, b, 0, 'aug_before', color='blue')
                self.debug.plot_state_rviz(inputs_aug, b, 1, 'aug_after', color='blue')
                self.debug.plot_action_rviz(inputs_aug, b, 'aug', color='blue')

        on_invalid_aug = self.hparams.get('on_invalid_aug', 'original')
        if on_invalid_aug == 'original':
            inputs_aug, local_env_aug, local_origin_point_aug = self.use_original_if_invalid(is_valid, batch_size,
                                                                                             inputs,
                                                                                             inputs_aug, local_env,
                                                                                             local_env_aug,
                                                                                             local_origin_point,
                                                                                             local_origin_point_aug)
        elif on_invalid_aug == 'drop':
            if tf.reduce_any(tf.cast(is_valid, tf.bool)):
                inputs_aug, local_env_aug, local_origin_point_aug = self.drop_if_invalid(is_valid, batch_size,
                                                                                         None, inputs_aug,
                                                                                         None, local_env_aug,
                                                                                         None, local_origin_point_aug)
            else:
                print("All augmentations in the batch are invalid!")
                inputs_aug, local_env_aug, local_origin_point_aug = self.use_original_if_invalid(is_valid, batch_size,
                                                                                                 inputs, inputs_aug,
                                                                                                 local_env,
                                                                                                 local_env_aug,
                                                                                                 local_origin_point,
                                                                                                 local_origin_point_aug)
        else:
            raise NotImplementedError(on_invalid_aug)

        return inputs_aug, local_env_aug, local_origin_point_aug

    def drop_if_invalid(self, is_valid, batch_size,
                        _, inputs_aug,
                        __, local_env_aug,
                        ___, local_origin_point_aug):
        valid_indices = tf.squeeze(tf.where(is_valid), axis=-1)
        n_tile = tf.cast(tf.cast(batch_size, tf.int64) / tf.cast(tf.size(valid_indices), tf.int64), tf.int64) + 1
        repeated_indices = tf.tile(valid_indices, [n_tile])[:batch_size]  # ex: [0,19,22], 8 --> [0,19,22,0,19,22,0,19]
        inputs_aug_valid = {}
        for k, v in inputs_aug.items():
            if k in ['batch_size', 'time']:
                inputs_aug_valid[k] = v
            else:
                inputs_aug_valid[k] = tf.gather(v, repeated_indices, axis=0)
        local_env_aug = tf.gather(local_env_aug, repeated_indices, axis=0)
        local_origin_point_aug = tf.gather(local_origin_point_aug, repeated_indices, axis=0)
        return inputs_aug_valid, local_env_aug, local_origin_point_aug

    def use_original_if_invalid(self, is_valid, batch_size, inputs, inputs_aug, local_env, local_env_aug,
                                local_origin_point, local_origin_point_aug):
        # FIXME: this is hacky
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
        n_sample = 1000
        transformation_params = self.scenario.sample_object_augmentation_variables(n_sample, self.seed)
        best_transformation_params = pick_best_params(self, batch_size, transformation_params)
        transformation_matrices = [transformations.compose_matrix(translate=p[:3], angles=p[3:]) for p in
                                   best_transformation_params]
        return tf.cast(transformation_matrices, tf.float32)

    def apply_object_augmentation_no_ik(self, transformation_matrices, to_local_frame, inputs, batch_size, time):
        return self.scenario.apply_object_augmentation_no_ik(transformation_matrices,
                                                             to_local_frame,
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

    def sample_initial_transforms(self, batch_size):
        # Sample an initial random object transformation. This can be the same across the batch
        n_sample = 1000
        initial_transformation_params = self.scenario.sample_object_augmentation_variables(n_sample, self.seed)
        # pick the most valid transforms, via the learned object state augmentation validity model
        initial_transformation_params = pick_best_params(self, batch_size, initial_transformation_params)
        return initial_transformation_params

    def can_terminate(self, lr, gradients):
        grad_norm = tf.linalg.norm(gradients[0], axis=-1)
        step_size_i = grad_norm * lr
        can_terminate = step_size_i < self.step_size_threshold
        can_terminate = tf.reduce_all(can_terminate)
        return can_terminate

    def clip_env_aug_grad(self, gradients, variables):
        def _clip(g):
            # we want grad_clip to be as close to in meters as possible, so here we scale by step size
            c = self.grad_clip / self.hparams['step_size']
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
        if add_new('sdf') in example:
            new_env['sdf'] = example[add_new('sdf')]
        if add_new('sdf_grad') in example:
            new_env['sdf_grad'] = example[add_new('sdf_grad')]
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

    def bbox_loss(self, obj_points_aug, extent):
        extent = tf.reshape(extent, [3, 2])
        lower_extent = extent[None, None, :, 0]
        upper_extent = extent[None, None, :, 1]
        lower_extent_loss = tf.maximum(0., obj_points_aug - upper_extent)
        upper_extent_loss = tf.maximum(0., lower_extent - obj_points_aug)
        bbox_loss = tf.reduce_sum(lower_extent_loss + upper_extent_loss, axis=-1)
        return self.hparams['bbox_weight'] * bbox_loss
