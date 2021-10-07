import pathlib
from typing import Dict, List

import tensorflow as tf
import tensorflow_probability as tfp
from pyjacobian_follower import IkParams

import rospy
from arm_robots.robot_utils import merge_joint_state_and_scene_msg
from learn_invariance.invariance_model_wrapper import InvarianceModelWrapper
from link_bot_classifiers.aug_opt_ik import AugOptIk
from link_bot_classifiers.aug_opt_utils import debug_aug, debug_ik, check_env_constraints, pick_best_params, \
    initial_identity_params, transform_obj_points
from link_bot_classifiers.aug_projection_opt import AugProjOpt
from link_bot_classifiers.classifier_debugging import ClassifierDebugging
from link_bot_classifiers.iterative_projection import iterative_projection
from link_bot_classifiers.local_env_helper import LocalEnvHelper
from link_bot_classifiers.make_voxelgrid_inputs import VoxelgridInfo
from link_bot_data.dataset_utils import add_new, add_predicted, deserialize_scene_msg
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from link_bot_pycommon.grid_utils import lookup_points_in_vg, send_voxelgrid_tf_origin_point_res, environment_to_vg_msg
from link_bot_pycommon.pycommon import densify_points
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.geometry import transformation_params_to_matrices
from moonshine.moonshine_utils import to_list_of_strings, possibly_none_concat, repeat
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

        self.seed_int = 4 if self.hparams is None or 'seed' not in self.hparams else self.hparams['seed']
        self.gen = tf.random.Generator.from_seed(self.seed_int)
        self.seed = tfp.util.SeedStream(self.seed_int + 1, salt="nn_classifier_aug")
        self.ik_solver = None

        if self.do_augmentation():
            ik_params = IkParams(rng_dist=self.hparams.get("rand_dist", 0.1),
                                 max_collision_check_attempts=self.hparams.get("max_collision_check_attempts", 1))
            self.ik_solver = AugOptIk(scenario.robot, ik_params=ik_params)

            invariance_model_path = pathlib.Path(self.hparams['invariance_model'])
            self.invariance_model_wrapper = InvarianceModelWrapper(invariance_model_path, self.batch_size,
                                                                   self.scenario)

        # metrics

    def augmentation_optimization(self,
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
        inputs_aug, local_origin_point_aug, local_center_aug, local_env_aug, is_obj_aug_valid = \
            self.opt_object_augmentation(inputs,
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

        inputs_aug, local_env_aug, local_origin_point_aug = self.use_original_if_invalid(is_valid, batch_size,
                                                                                         inputs,
                                                                                         inputs_aug, local_env,
                                                                                         local_env_aug,
                                                                                         local_origin_point,
                                                                                         local_origin_point_aug)

        # add more useful info
        inputs_aug['is_valid'] = is_valid

        return inputs_aug, local_env_aug, local_origin_point_aug

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
            for b in debug_viz_batch_indices(batch_size):
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

        initial_transformation_params = initial_identity_params(batch_size)
        target_transformation_params = self.sample_target_transform_params(batch_size)
        project_opt = AugProjOpt(aug_opt=self,
                                 new_env=new_env,
                                 res=res,
                                 batch_size=batch_size,
                                 obj_points=obj_points,
                                 object_points_occupancy=object_points_occupancy)
        not_progressing_threshold = self.hparams['not_progressing_threshold']
        obj_transforms, viz_vars = iterative_projection(initial_value=initial_transformation_params,
                                                        target=target_transformation_params,
                                                        n=self.hparams['n_outer_iters'],
                                                        m=self.hparams['max_steps'],
                                                        step_towards_target=project_opt.step_towards_target,
                                                        project_opt=project_opt,
                                                        x_distance=project_opt.distance,
                                                        not_progressing_threshold=not_progressing_threshold,
                                                        viz_func=project_opt.viz_func,
                                                        viz=debug_aug())

        transformation_matrices = transformation_params_to_matrices(obj_transforms, batch_size)
        obj_points_aug, to_local_frame = transform_obj_points(obj_points, transformation_matrices)

        # this updates other representations of state/action that are fed into the network
        _, object_aug_update, local_origin_point_aug, local_center_aug = self.apply_object_augmentation_no_ik(
            transformation_matrices,
            to_local_frame,
            inputs,
            batch_size,
            time)
        inputs_aug.update(object_aug_update)

        if debug_aug():
            for b in debug_viz_batch_indices(batch_size):
                self.debug.send_position_transform(local_origin_point_aug[b], 'local_origin_point_aug')
                self.debug.send_position_transform(local_center_aug[b], 'local_center_aug')

        new_env_repeated = repeat(new_env, repetitions=batch_size, axis=0, new_axis=True)
        local_env_aug, _ = self.local_env_helper.get(local_center_aug, new_env_repeated, batch_size)

        is_valid = self.check_is_valid(obj_points_aug, new_env, object_points_occupancy, res, project_opt.obj_sdf,
                                       viz_vars.sdf_aug)

        return inputs_aug, local_origin_point_aug, local_center_aug, local_env_aug, is_valid

    def check_is_valid(self, obj_points_aug, new_env, attract_mask, res, sdf_dist, sdf_dist_aug):
        bbox_loss_batch = self.bbox_loss(obj_points_aug, new_env['extent'])
        bbox_constraint_satisfied = tf.cast(tf.reduce_sum(bbox_loss_batch, axis=-1) == 0, tf.float32)

        env_constraints_satisfied_ = check_env_constraints(attract_mask, sdf_dist_aug, res)
        num_env_constraints_violated = tf.reduce_sum(1 - env_constraints_satisfied_, axis=1)
        env_constraints_satisfied = tf.cast(num_env_constraints_violated < self.hparams['max_env_violations'],
                                            tf.float32)

        min_dist = tf.reduce_min(sdf_dist, axis=1)
        min_dist_aug = tf.reduce_min(sdf_dist_aug, axis=1)
        delta_min_dist = tf.abs(min_dist - min_dist_aug)
        delta_min_dist_satisfied = tf.cast(delta_min_dist < self.hparams['delta_min_dist_threshold'], tf.float32)

        constraints_satisfied = env_constraints_satisfied * bbox_constraint_satisfied * delta_min_dist_satisfied
        return constraints_satisfied

    def sample_target_transform_params(self, batch_size):
        good_enough_percentile = self.hparams['good_enough_percentile']
        n_samples = int(1 / good_enough_percentile) * batch_size

        trans_lim = tf.ones([3]) * self.hparams['target_trans_lim']
        trans_distribution = tfp.distributions.Uniform(low=-trans_lim, high=trans_lim)

        # NOTE: by restricting the sample of euler angles to < pi/2 we can ensure that the representation is unique.
        #  (see https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf) which allows us to use a simple distance euler
        #  function between two sets of euler angles.
        euler_lim = tf.ones([3]) * self.hparams['target_euler_lim']
        euler_distribution = tfp.distributions.Uniform(low=-euler_lim, high=euler_lim)

        trans_target = trans_distribution.sample(sample_shape=n_samples, seed=self.seed())
        euler_target = euler_distribution.sample(sample_shape=n_samples, seed=self.seed())

        target_params = tf.concat([trans_target, euler_target], 1)

        # pick the most valid transforms, via the learned object state augmentation validity model
        best_target_params = pick_best_params(self, batch_size, target_params)
        return best_target_params

    def use_original_if_invalid(self,
                                is_valid,
                                batch_size,
                                inputs,
                                inputs_aug,
                                local_env,
                                local_env_aug,
                                local_origin_point,
                                local_origin_point_aug):
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
            'batch_size':           batch_size,
            'time':                 inputs['time'],
            add_predicted('stdev'): inputs[add_predicted('stdev')],
        }

        local_env, local_origin_point = self.get_local_env(inputs, batch_size)

        object_points = self.compute_swept_object_points(inputs, batch_size)
        res = inputs['res']
        # get all components of the state as a set of points
        # in general this should be the swept volume, and should include the robot
        object_points_occupancy = lookup_points_in_vg(object_points, local_env, res, local_origin_point, batch_size)
        return inputs_aug, res, object_points, object_points_occupancy, local_env, local_origin_point

    def apply_object_augmentation_no_ik(self, transformation_matrices, to_local_frame, inputs, batch_size, time):
        return self.scenario.apply_object_augmentation_no_ik(transformation_matrices,
                                                             to_local_frame,
                                                             inputs,
                                                             batch_size,
                                                             time,
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

    def do_augmentation(self):
        return self.hparams is not None

    def compute_swept_object_points(self, inputs, batch_size):
        points_state_keys = [add_predicted(k) for k in self.points_state_keys]

        def _make_points(k, t):
            v = inputs[k][:, t]
            points = tf.reshape(v, [batch_size, -1, 3])
            points = densify_points(batch_size, points)
            return points

        object_points_0 = {k: _make_points(k, 0) for k in points_state_keys}
        object_points_1 = {k: _make_points(k, 1) for k in points_state_keys}

        def _linspace(k):
            return tf.linspace(object_points_0[k], object_points_1[k], self.hparams['num_object_interp'], axis=1)

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

    def bbox_loss(self, obj_points_aug, extent):
        extent = tf.reshape(extent, [3, 2])
        lower_extent = extent[None, None, :, 0]
        upper_extent = extent[None, None, :, 1]
        lower_extent_loss = tf.maximum(0., obj_points_aug - upper_extent)
        upper_extent_loss = tf.maximum(0., lower_extent - obj_points_aug)
        bbox_loss = tf.reduce_sum(lower_extent_loss + upper_extent_loss, axis=-1)
        return self.hparams['bbox_weight'] * bbox_loss
