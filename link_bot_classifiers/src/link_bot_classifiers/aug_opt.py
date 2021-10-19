import pathlib
from typing import Dict, List

import tensorflow as tf
import tensorflow_probability as tfp
from pyjacobian_follower import IkParams

import sdf_tools.utils_3d_tensorflow
from arm_robots.robot_utils import merge_joint_state_and_scene_msg
from learn_invariance.invariance_model_wrapper import InvarianceModelWrapper
from link_bot_classifiers.aug_opt_ik import AugOptIk
from link_bot_classifiers.aug_opt_utils import debug_aug, debug_input, debug_ik, check_env_constraints, \
    pick_best_params, \
    initial_identity_params, transform_obj_points
from link_bot_classifiers.aug_projection_opt import AugProjOpt
from link_bot_classifiers.classifier_debugging import ClassifierDebugging
from link_bot_classifiers.iterative_projection import iterative_projection
from link_bot_classifiers.local_env_helper import LocalEnvHelper
from link_bot_classifiers.make_voxelgrid_inputs import VoxelgridInfo
from link_bot_data.dataset_utils import add_predicted, deserialize_scene_msg
from link_bot_data.visualization_common import make_delete_marker, make_delete_markerarray
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from link_bot_pycommon.grid_utils import lookup_points_in_vg
from link_bot_pycommon.pycommon import densify_points
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.geometry import transformation_params_to_matrices
from moonshine.moonshine_utils import to_list_of_strings
from sdf_tools import utils_3d
from sensor_msgs.msg import JointState
from visualization_msgs.msg import MarkerArray

cache_ = {}


def sdf_and_grad_cached(env, res, origin_point, batch_size):
    global cache_

    if isinstance(env, tf.Tensor):
        key = env.numpy().tostring() + res.numpy().tostring() + origin_point.numpy().tostring()
    else:
        key = env.tostring() + str(res).encode("utf-8") + origin_point.tostring()
    if key in cache_:
        return cache_[key]
    print("Computing SDF, slow!!!", len(cache_))
    v = sdf_tools.utils_3d_tensorflow.compute_sdf_and_gradient_batch(env, res, origin_point, batch_size)
    cache_[key] = v
    return v


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

    def aug_opt(self, inputs: Dict, batch_size: int, time: int):
        if debug_aug():
            self.delete_state_action_markers()

        if 'sdf' not in inputs or 'sdf_grad' not in inputs:
            sdf, sdf_grad = sdf_and_grad_cached(inputs['env'], inputs['res'], inputs['origin_point'], batch_size)
            inputs['sdf'] = sdf
            inputs['sdf_grad'] = sdf_grad

        res = inputs['res']
        extent = inputs['extent']
        origin_point = inputs['origin_point']
        env = inputs['env']

        obj_points = self.compute_swept_obj_points(inputs, batch_size)
        # get all components of the state as a set of points. this could be the swept volume and/or include the robot
        obj_occupancy = lookup_points_in_vg(obj_points, env, res, origin_point, batch_size)

        transformation_matrices, to_local_frame, is_obj_aug_valid = self.aug_obj_transform(
            res=res,
            extent=extent,
            origin_point=origin_point,
            sdf=inputs['sdf'],
            sdf_grad=inputs['sdf_grad'],
            obj_points=obj_points,
            obj_occupancy=obj_occupancy,
            batch_size=batch_size)

        # apply the transformations to some components of the state/action
        obj_aug_update, local_origin_point_aug, local_center_aug = self.apply_object_augmentation_no_ik(
            transformation_matrices,
            to_local_frame,
            inputs,
            batch_size,
            time)

        # things that we won't be updating in this augmentation
        inputs_aug = {
            'res':                  res,
            'extent':               extent,
            'origin_point':         inputs['origin_point'],
            'env':                  inputs['env'],
            'sdf':                  inputs['sdf'],
            'sdf_grad':             inputs['sdf_grad'],
            'scene_msg':            inputs['scene_msg'],
            'is_close':             inputs['is_close'],
            'batch_size':           batch_size,
            'time':                 inputs['time'],
            'joint_names':          inputs['joint_names'],
            add_predicted('stdev'): inputs[add_predicted('stdev')],
            'error':                inputs['error'],
        }
        inputs_aug.update(obj_aug_update)

        if debug_input():
            for b in debug_viz_batch_indices(batch_size):
                self.debug.send_position_transform(local_origin_point_aug[b], 'local_origin_point_aug')
                self.debug.send_position_transform(local_center_aug[b], 'local_center_aug')
                self.debug.plot_state_rviz(inputs_aug, b, 0, 'aug_before', color='blue')
                self.debug.plot_state_rviz(inputs_aug, b, 1, 'aug_after', color='blue')
                self.debug.plot_action_rviz(inputs_aug, b, 'aug', color='blue')
                env_b = {
                    'env':          env[b].numpy(),
                    'res':          res[b].numpy(),
                    'origin_point': origin_point[b].numpy(),
                    'extent':       extent[b].numpy(),
                }
                self.scenario.plot_environment_rviz(env_b)
                self.debug.send_position_transform(origin_point[b], 'origin_point')

        default_robot_positions = inputs[add_predicted('joint_positions')][:, 0]
        joint_positions_aug, is_ik_valid = self.solve_ik(inputs_aug, default_robot_positions, batch_size)
        inputs_aug.update({
            add_predicted('joint_positions'): joint_positions_aug,
            'joint_names':                    inputs['joint_names'],
        })

        is_valid = is_ik_valid * is_obj_aug_valid

        inputs_aug = self.use_original_if_invalid(is_valid, batch_size, inputs, inputs_aug)

        # add some more useful info
        inputs_aug['is_valid'] = is_valid

        return inputs_aug

    def aug_obj_transform(self,
                          res,
                          extent,
                          origin_point,
                          sdf,
                          sdf_grad,
                          obj_points,
                          obj_occupancy,
                          batch_size: int,
                          ):
        initial_transformation_params = initial_identity_params(batch_size)
        target_transformation_params = self.sample_target_transform_params(batch_size)
        project_opt = AugProjOpt(aug_opt=self,
                                 sdf=sdf,
                                 sdf_grad=sdf_grad,
                                 res=res,
                                 origin_point=origin_point,
                                 extent=extent,
                                 batch_size=batch_size,
                                 obj_points=obj_points,
                                 obj_occupancy=obj_occupancy)
        if debug_aug():
            project_opt.clear_viz()
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

        is_valid = self.check_is_valid(obj_points_aug=obj_points_aug,
                                       obj_occupancy=obj_occupancy,
                                       extent=extent,
                                       res=res,
                                       sdf=project_opt.obj_sdf,
                                       sdf_aug=viz_vars.sdf_aug)

        return transformation_matrices, to_local_frame, is_valid

    def check_is_valid(self, obj_points_aug, obj_occupancy, extent, res, sdf, sdf_aug):
        bbox_loss_batch = self.bbox_loss(obj_points_aug, extent)
        bbox_constraint_satisfied = tf.cast(tf.reduce_sum(bbox_loss_batch, axis=-1) == 0, tf.float32)

        env_constraints_satisfied_ = check_env_constraints(obj_occupancy, sdf_aug, res)
        num_env_constraints_violated = tf.reduce_sum(1 - env_constraints_satisfied_, axis=1)
        env_constraints_satisfied = tf.cast(num_env_constraints_violated < self.hparams['max_env_violations'],
                                            tf.float32)

        min_dist = tf.reduce_min(sdf, axis=1)
        min_dist_aug = tf.reduce_min(sdf_aug, axis=1)
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
                                inputs_aug):
        # FIXME: this is hacky
        keys_aug = [add_predicted('joint_positions')]
        keys_aug += self.action_keys
        keys_aug += [add_predicted(psk) for psk in self.points_state_keys]
        for k in keys_aug:
            v = inputs_aug[k]
            iv = tf.reshape(is_valid, [batch_size] + [1] * (v.ndim - 1))
            inputs_aug[k] = iv * inputs_aug[k] + (1 - iv) * inputs[k]
        return inputs_aug

    def apply_object_augmentation_no_ik(self, transformation_matrices, to_local_frame, inputs, batch_size, time):
        return self.scenario.apply_object_augmentation_no_ik(transformation_matrices,
                                                             to_local_frame,
                                                             inputs,
                                                             batch_size,
                                                             time,
                                                             self.local_env_helper.h,
                                                             self.local_env_helper.w,
                                                             self.local_env_helper.c)

    def do_augmentation(self):
        return self.hparams is not None

    def compute_swept_obj_points(self, inputs, batch_size):
        points_state_keys = [add_predicted(k) for k in self.points_state_keys]

        def _make_points(k, t):
            v = inputs[k][:, t]
            points = tf.reshape(v, [batch_size, -1, 3])
            points = densify_points(batch_size, points)
            return points

        obj_points_0 = {k: _make_points(k, 0) for k in points_state_keys}
        obj_points_1 = {k: _make_points(k, 1) for k in points_state_keys}

        def _linspace(k):
            return tf.linspace(obj_points_0[k], obj_points_1[k], self.hparams['num_object_interp'], axis=1)

        swept_obj_points = tf.concat([_linspace(k) for k in points_state_keys], axis=2)
        swept_obj_points = tf.reshape(swept_obj_points, [batch_size, -1, 3])

        return swept_obj_points

    def solve_ik(self, inputs_aug: Dict, default_robot_positions, batch_size: int):
        left_gripper_points_aug = inputs_aug[add_predicted('left_gripper')]
        right_gripper_points_aug = inputs_aug[add_predicted('right_gripper')]
        deserialize_scene_msg(inputs_aug)

        # run ik, try to find collision free solution
        scene_msg = inputs_aug['scene_msg']
        joint_names = to_list_of_strings(inputs_aug['joint_names'][0, 0].numpy().tolist())
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
        extent = tf.reshape(extent, [-1, 3, 2])
        lower_extent = extent[:, None, :, 0]
        upper_extent = extent[:, None, :, 1]
        lower_extent_loss = tf.maximum(0., obj_points_aug - upper_extent)
        upper_extent_loss = tf.maximum(0., lower_extent - obj_points_aug)
        bbox_loss = tf.reduce_sum(lower_extent_loss + upper_extent_loss, axis=-1)
        return self.hparams['bbox_weight'] * bbox_loss

    def delete_state_action_markers(self):
        label = 'aug'
        state_delete_msg = MarkerArray(markers=[make_delete_marker(ns=label + '_l'),
                                                make_delete_marker(ns=label + '_r'),
                                                make_delete_marker(ns=label + '_rope')])
        self.scenario.state_viz_pub.publish(state_delete_msg)
        action_delete_msg = MarkerArray(markers=[make_delete_marker(ns=label)])
        self.scenario.action_viz_pub.publish(action_delete_msg)
        self.scenario.arrows_pub.publish(make_delete_markerarray())
        self.scenario.arrows_pub.publish(make_delete_markerarray())
