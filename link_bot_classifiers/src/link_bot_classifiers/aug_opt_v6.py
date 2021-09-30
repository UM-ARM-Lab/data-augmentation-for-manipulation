from typing import Dict

import tensorflow as tf
import tensorflow_probability as tfp

import rospy
from link_bot_classifiers.aug_opt_utils import debug_aug, debug_aug_sgd, transformation_obj_points, \
    check_env_constraints, pick_best_params, initial_identity_params
from link_bot_classifiers.iterative_projection import iterative_projection, BaseProjectOpt
from link_bot_data.rviz_arrow import rviz_arrow
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from link_bot_pycommon.grid_utils import environment_to_vg_msg, send_voxelgrid_tf_origin_point_res, \
    batch_point_to_idx
from moonshine.geometry import transformation_params_to_matrices, transformation_jacobian, \
    homogeneous, quat_dist
from moonshine.moonshine_utils import repeat
from sdf_tools import utils_3d
from tf import transformations
from visualization_msgs.msg import Marker


def delta_min_dist_loss(sdf_dist, sdf_dist_aug):
    min_dist = tf.reduce_min(sdf_dist, axis=1)
    min_dist_aug = tf.reduce_min(sdf_dist_aug, axis=1)
    delta_min_dist = tf.abs(min_dist - min_dist_aug)
    return delta_min_dist


def dpoint_to_dparams(dpoint, dpoint_dparams):
    dparams = tf.einsum('bni,bnij->bnj', dpoint, dpoint_dparams)  # [b,n,6]
    dparams = tf.reduce_mean(dparams, axis=1)
    return dparams


class AugV6ProjOpt(BaseProjectOpt):
    def __init__(self, aug_opt, new_env, res, batch_size, obj_points, object_points_occupancy):
        super().__init__()
        self.aug_opt = aug_opt
        self.new_env = new_env
        self.new_res = self.new_env['res']
        self.new_origin_point = self.new_env['origin_point']
        self.new_origin_point_expanded = self.new_env['origin_point']
        self.res = res
        self.batch_size = batch_size
        self.obj_points = obj_points
        self.object_points_occupancy = object_points_occupancy

        # More hyperparameters
        self.step_toward_target_fraction = 1 / self.aug_opt.hparams['n_outer_iters']
        self.lr_decay = 0.90
        self.lr_decay_steps = 10
        self.attract_weight = 15.0 * self.aug_opt.hparams['sdf_grad_scale']
        self.repel_weight = 1.0 * self.aug_opt.hparams['sdf_grad_scale']

        self.aug_dir_pub = rospy.Publisher('aug_dir', Marker, queue_size=10)

        if 'sdf' in self.new_env and 'sdf_grad' in self.new_env:
            sdf = self.new_env['sdf']
            sdf_grad = self.new_env['sdf_grad']
        else:
            print("Computing SDF online, very slow!")
            sdf, sdf_grad = utils_3d.compute_sdf_and_gradient(self.new_env['env'],
                                                              self.new_res,
                                                              self.new_origin_point)

        self.sdf = tf.convert_to_tensor(sdf)
        self.sdf_grad = tf.convert_to_tensor(sdf_grad)
        repel_grad_mask = tf.cast(sdf < self.aug_opt.barrier_cut_off, tf.float32)
        self.repel_sdf_grad = self.sdf_grad * tf.expand_dims(repel_grad_mask, -1)
        attract_grad_mask = tf.cast(sdf > 0, tf.float32)
        self.attract_sdf_grad = self.sdf_grad * tf.expand_dims(attract_grad_mask, -1)

    def make_opt(self):
        lr = tf.keras.optimizers.schedules.ExponentialDecay(self.aug_opt.hparams['step_size'],
                                                            self.aug_opt.hparams['lr_decay_steps'],
                                                            self.aug_opt.hparams['lr_decay'])
        opt = tf.keras.optimizers.SGD(lr)
        return opt

    def forward(self, tape, obj_transforms):
        with tape:
            # obj_points is the set of points that define the object state, ie. the swept rope points.
            # the obj_points are in world frame, so obj_params is a transform in world frame
            # to compute the object state constraints loss we need to transform this during each forward pass
            # we also need to call apply_object_augmentation* at the end
            # to update the rest of the "state" which is input to the network
            transformation_matrices = transformation_params_to_matrices(obj_transforms, self.batch_size)
            obj_points_aug, to_local_frame = transformation_obj_points(self.obj_points, transformation_matrices)

        # compute repel and attract gradient via the SDF
        obj_point_indices = batch_point_to_idx(self.obj_points, self.new_res, self.new_origin_point_expanded)
        sdf_dist = tf.gather_nd(self.new_env['sdf'], obj_point_indices)  # will be zero if index OOB

        obj_point_indices_aug = batch_point_to_idx(obj_points_aug, self.new_res, self.new_origin_point_expanded)
        sdf_dist_aug = tf.gather_nd(self.sdf, obj_point_indices_aug)  # will be zero if index OOB
        obj_attract_sdf_grad = tf.gather_nd(self.attract_sdf_grad,
                                            obj_point_indices_aug)  # will be zero if index OOB
        obj_repel_sdf_grad = tf.gather_nd(self.repel_sdf_grad, obj_point_indices_aug)  # will be zero if index OOB

        # and also the grad for preserving the min dist
        min_dist = tf.reduce_min(sdf_dist, axis=1)
        min_dist_aug = tf.reduce_min(sdf_dist_aug, axis=1)
        min_dist_aug_idx = tf.argmin(sdf_dist_aug, axis=1)
        delta_min_dist = min_dist - min_dist_aug
        min_dist_points_aug = tf.gather(obj_points_aug, min_dist_aug_idx, axis=1, batch_dims=1)
        min_dist_indices_aug = batch_point_to_idx(min_dist_points_aug, self.new_res, self.new_origin_point_expanded)
        delta_min_dist_grad_dpoint = -tf.gather_nd(self.sdf_grad, min_dist_indices_aug) * delta_min_dist[:, None]
        delta_min_dist_grad_dpoint = delta_min_dist_grad_dpoint * self.aug_opt.hparams['delta_min_dist_weight']

        return obj_attract_sdf_grad, obj_repel_sdf_grad, sdf_dist, sdf_dist_aug, obj_point_indices_aug, obj_points_aug, to_local_frame, min_dist_points_aug, delta_min_dist_grad_dpoint, min_dist_aug_idx

    def step(self, _: int, opt, obj_transforms: tf.Variable):
        tape = tf.GradientTape()

        viz_vars = self.forward(tape, obj_transforms)
        obj_attract_sdf_grad, obj_repel_sdf_grad, sdf_dist, sdf_dist_aug, obj_point_indices_aug, obj_points_aug, to_local_frame, min_dist_points_aug, delta_min_dist_grad_dpoint, min_dist_aug_idx = viz_vars

        with tape:
            invariance_loss = self.aug_opt.invariance_model_wrapper.evaluate(obj_transforms)
            invariance_loss = tf.maximum(self.aug_opt.hparams['invariance_threshold'], invariance_loss)
            invariance_loss = self.aug_opt.hparams['invariance_weight'] * invariance_loss

            bbox_loss_batch = self.aug_opt.bbox_loss(obj_points_aug, self.new_env['extent'])
            bbox_loss = tf.reduce_sum(bbox_loss_batch, axis=-1)

            losses = [
                bbox_loss,
                invariance_loss,
            ]
            losses_sum = tf.add_n(losses)
            loss = tf.reduce_mean(losses_sum)

        attract_mask = self.object_points_occupancy  # assumed to already be either 0.0 or 1.0
        attract_grad = obj_attract_sdf_grad * tf.expand_dims(attract_mask, -1) * self.attract_weight
        repel_grad = -obj_repel_sdf_grad * tf.expand_dims((1 - attract_mask), -1) * self.repel_weight
        attract_repel_dpoint = (attract_grad + repel_grad)  # [b,n,3]

        # Compute the jacobian of the transformation
        jacobian = transformation_jacobian(obj_transforms)[:, None]  # [b,1,6,4,4]
        obj_points_local_frame = self.obj_points - to_local_frame  # [b,n,3]
        obj_points_local_frame_h = homogeneous(obj_points_local_frame)[:, :, None, :, None]  # [b,1,4,1]
        dpoint_dparams_h = tf.squeeze(tf.matmul(jacobian, obj_points_local_frame_h), axis=-1)  # [b,6]
        dpoint_dparams = tf.transpose(dpoint_dparams_h[:, :, :, :3], [0, 1, 3, 2])  # [b,3,6]

        # chain rule
        attract_repel_sdf_grad = dpoint_to_dparams(attract_repel_dpoint, dpoint_dparams)
        dpoint_dparams_for_min_point = tf.gather(dpoint_dparams, min_dist_aug_idx, axis=1, batch_dims=1)
        delta_min_dist_grad_dparams = tf.einsum('bi,bij->bj', delta_min_dist_grad_dpoint, dpoint_dparams_for_min_point)

        variables = [obj_transforms]
        tape_gradients = tape.gradient(loss, variables)

        # combine with the gradient for the other aspects of the loss, those computed by tf.gradient
        gradients = [tape_gradients[0] + attract_repel_sdf_grad + delta_min_dist_grad_dparams]

        clipped_grads_and_vars = self.aug_opt.clip_env_aug_grad(gradients, variables)
        opt.apply_gradients(grads_and_vars=clipped_grads_and_vars)
        lr = opt._decayed_lr(tf.float32)
        can_terminate = self.aug_opt.can_terminate(lr, gradients)

        x_out = tf.convert_to_tensor(obj_transforms)

        return x_out, can_terminate, viz_vars

    def step_towards_target(self, target, obj_transforms):
        euler_interp = []
        trans_interp = []
        # TODO: batch this
        for b in range(self.batch_size):
            x_b = obj_transforms[b].numpy()
            target_b = target[b].numpy()
            q1_b = transformations.quaternion_from_euler(*x_b[3:])
            q2_b = transformations.quaternion_from_euler(*target_b[3:])
            q_interp_b = transformations.quaternion_slerp(q1_b, q2_b, self.step_toward_target_fraction)
            trans1_b = x_b[:3]
            trans2_b = target_b[:3]
            trans_interp_b = trans1_b + (trans2_b - trans1_b) * self.step_toward_target_fraction
            euler_interp_b = transformations.euler_from_quaternion(q_interp_b)
            euler_interp.append(tf.convert_to_tensor(euler_interp_b, tf.float32))
            trans_interp.append(trans_interp_b)
        euler_interp = tf.stack(euler_interp, axis=0)
        trans_interp = tf.stack(trans_interp, axis=0)
        x_interp = tf.concat((trans_interp, euler_interp), axis=1)

        tape = tf.GradientTape()
        viz_vars = self.forward(tape, x_interp)
        return x_interp, viz_vars

    def distance(self, transforms1, transforms2):
        distances = []
        # TODO: batch this
        for b in range(self.batch_size):
            transforms1_b = transforms1[b]
            transforms2_b = transforms2[b]
            euler1_b = transforms1_b[3:]
            euler2_b = transforms2_b[3:]
            quat1_b = transformations.quaternion_from_euler(*euler1_b.numpy())
            quat2_b = transformations.quaternion_from_euler(*euler2_b.numpy())
            quat_dist_b = tf.cast(quat_dist(quat1_b, quat2_b), tf.float32)
            trans1_b = transforms1_b[:3]
            trans2_b = transforms2_b[:3]
            trans_dist_b = tf.linalg.norm(trans1_b - trans2_b)
            distance_b = quat_dist_b + trans_dist_b
            distances.append(distance_b)
        max_distance = tf.reduce_max(distances)
        return max_distance

    def viz_func(self, i, obj_transforms, _, target, viz_vars):
        obj_attract_sdf_grad, obj_repel_sdf_grad, sdf_dist, sdf_dist_aug, obj_point_indices_aug, obj_points_aug, to_local_frame, min_dist_points_aug, delta_min_dist_grad_dpoint, min_dist_aug_idx = viz_vars

        shape = tf.convert_to_tensor(self.new_env['env'].shape, tf.int64)[None, None]
        oob = tf.logical_or(obj_point_indices_aug < 0,
                            obj_point_indices_aug > shape)
        oob = tf.reduce_any(oob, axis=-1)

        if debug_aug_sgd():

            scale = 0.005
            for b in debug_viz_batch_indices(self.batch_size):
                target_b = target[b]
                obj_transforms_b = obj_transforms[b]

                local_pos_b = to_local_frame[b, 0].numpy()
                target_pos_b = target_b[:3].numpy()
                self.aug_opt.debug.send_position_transform(local_pos_b, 'initial_local_frame')
                self.plot_transform(obj_transforms_b, 'aug_opt_current')
                self.plot_transform(target_b, 'aug_opt_target')

                dir_msg = rviz_arrow([0, 0, 0], target_pos_b, scale=2.0)
                dir_msg.header.frame_id = 'initial_local_frame'
                self.aug_dir_pub.publish(dir_msg)

                repel_indices = tf.squeeze(tf.where(1 - self.object_points_occupancy[b]), -1)
                attract_indices = tf.squeeze(tf.where(self.object_points_occupancy[b]), -1)

                attract_points = tf.gather(self.obj_points[b], attract_indices).numpy()
                repel_points = tf.gather(self.obj_points[b], repel_indices).numpy()
                attract_points_aug = tf.gather(obj_points_aug[b], attract_indices).numpy()
                repel_points_aug = tf.gather(obj_points_aug[b], repel_indices).numpy()

                self.aug_opt.scenario.plot_points_rviz(attract_points, label='attract', color='g', scale=scale)
                self.aug_opt.scenario.plot_points_rviz(repel_points, label='repel', color='r', scale=scale)
                self.aug_opt.scenario.plot_points_rviz(attract_points_aug, label='attract_aug', color='g', scale=scale)
                self.aug_opt.scenario.plot_points_rviz(repel_points_aug, label='repel_aug', color='r', scale=scale)

                attract_grad_b = -tf.gather(obj_attract_sdf_grad[b], attract_indices, axis=0) * 0.02
                repel_sdf_dist = tf.gather(sdf_dist_aug[b], repel_indices, axis=0)
                repel_oob = tf.gather(oob[b], repel_indices, axis=0)
                repel_sdf_grad_b = tf.gather(obj_repel_sdf_grad[b], repel_indices, axis=0)
                repel_close = tf.logical_and(repel_sdf_dist < self.aug_opt.barrier_cut_off, tf.logical_not(repel_oob))
                repel_close_indices = tf.squeeze(tf.where(repel_close), axis=-1)
                repel_close_grad_b = tf.gather(repel_sdf_grad_b, repel_close_indices, axis=0) * 0.02
                repel_close_points_aug = tf.gather(repel_points_aug, repel_close_indices)
                self.aug_opt.scenario.delete_arrows_rviz(label='attract_sdf_grad')
                self.aug_opt.scenario.delete_arrows_rviz(label='repel_sdf_grad')
                self.aug_opt.scenario.plot_arrows_rviz(attract_points_aug, attract_grad_b, label='attract_sdf_grad',
                                                       color='g', scale=0.5)
                self.aug_opt.scenario.plot_arrows_rviz(repel_close_points_aug, repel_close_grad_b,
                                                       label='repel_sdf_grad',
                                                       color='r', scale=0.5)

                delta_min_dist_points_b = min_dist_points_aug[b].numpy()
                delta_min_dist_grad_b = delta_min_dist_grad_dpoint[b].numpy()
                self.aug_opt.scenario.plot_arrow_rviz(delta_min_dist_points_b, delta_min_dist_grad_b,
                                                      label='delta_min_dist_grad', color='pink', scale=0.5)
                # rospy.sleep(0.1)

    def plot_transform(self, transform_params, frame_id):
        """

        Args:
            transform_params: [x,y,z,roll,pitch,yaw]

        Returns:

        """
        target_pos_b = transform_params[:3].numpy()
        target_euler_b = transform_params[3:].numpy()
        target_q_b = transformations.quaternion_from_euler(*target_euler_b)
        self.aug_opt.scenario.tf.send_transform(target_pos_b, target_q_b, 'initial_local_frame', frame_id, False)


def opt_object_augmentation6(aug_opt,
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
            aug_opt.debug.env_aug_pub1.publish(msg)

            send_voxelgrid_tf_origin_point_res(aug_opt.broadcaster,
                                               origin_point=new_env['origin_point'],
                                               res=res[b],
                                               frame='new_env_aug_vg')

    initial_transformation_params = initial_identity_params(batch_size)
    target_transformation_params = sample_target_transform_params(aug_opt, obj_points, new_env, batch_size)
    project_opt = AugV6ProjOpt(aug_opt=aug_opt, new_env=new_env, res=res, batch_size=batch_size, obj_points=obj_points,
                               object_points_occupancy=object_points_occupancy)
    not_progressing_threshold = aug_opt.hparams['not_progressing_threshold']
    obj_transforms, viz_vars = iterative_projection(initial_value=initial_transformation_params,
                                                    target=target_transformation_params,
                                                    n=aug_opt.hparams['n_outer_iters'],
                                                    m=aug_opt.hparams['max_steps'],
                                                    step_towards_target=project_opt.step_towards_target,
                                                    project_opt=project_opt,
                                                    x_distance=project_opt.distance,
                                                    not_progressing_threshold=not_progressing_threshold,
                                                    viz_func=project_opt.viz_func)
    sdf_dist = viz_vars[2]
    sdf_dist_aug = viz_vars[3]

    transformation_matrices = transformation_params_to_matrices(obj_transforms, batch_size)
    obj_points_aug, to_local_frame = transformation_obj_points(obj_points, transformation_matrices)

    # this updates other representations of state/action that are fed into the network
    _, object_aug_update, local_origin_point_aug, local_center_aug = aug_opt.apply_object_augmentation_no_ik(
        transformation_matrices,
        to_local_frame,
        inputs,
        batch_size,
        time)
    inputs_aug.update(object_aug_update)

    if debug_aug():
        for b in debug_viz_batch_indices(batch_size):
            aug_opt.debug.send_position_transform(local_origin_point_aug[b], 'local_origin_point_aug')
            aug_opt.debug.send_position_transform(local_center_aug[b], 'local_center_aug')

    new_env_repeated = repeat(new_env, repetitions=batch_size, axis=0, new_axis=True)
    local_env_aug, _ = aug_opt.local_env_helper.get(local_center_aug, new_env_repeated, batch_size)

    is_valid = check_is_valid(aug_opt, obj_points_aug, new_env, object_points_occupancy, res, sdf_dist, sdf_dist_aug)

    return inputs_aug, local_origin_point_aug, local_center_aug, local_env_aug, is_valid


def check_is_valid(aug_opt, obj_points_aug, new_env, attract_mask, res, sdf_dist, sdf_dist_aug):
    bbox_loss_batch = aug_opt.bbox_loss(obj_points_aug, new_env['extent'])
    bbox_constraint_satisfied = tf.cast(tf.reduce_sum(bbox_loss_batch, axis=-1) == 0, tf.float32)

    env_constraints_satisfied_ = check_env_constraints(attract_mask, sdf_dist_aug, res)
    num_env_constraints_violated = tf.reduce_sum(1 - env_constraints_satisfied_, axis=1)
    env_constraints_satisfied = tf.cast(num_env_constraints_violated < aug_opt.hparams['max_env_violations'],
                                        tf.float32)

    min_dist = tf.reduce_min(sdf_dist, axis=1)
    min_dist_aug = tf.reduce_min(sdf_dist_aug, axis=1)
    delta_min_dist = tf.abs(min_dist - min_dist_aug)
    delta_min_dist_satisfied = tf.cast(delta_min_dist < aug_opt.hparams['delta_min_dist_threshold'], tf.float32)

    constraints_satisfied = env_constraints_satisfied * bbox_constraint_satisfied * delta_min_dist_satisfied
    return constraints_satisfied


def sample_target_transform_params(aug_opt, obj_points, new_env, batch_size):
    good_enough_percentile = aug_opt.hparams['good_enough_percentile']
    n_samples = int(1 / good_enough_percentile) * batch_size

    extent = tf.reshape(new_env['extent'], [3, 2])
    trans_low = extent[:, 0]
    trans_high = extent[:, 1]
    trans_distribution = tfp.distributions.Uniform(low=trans_low, high=trans_high)

    euler_low = tf.ones([3]) * -0.5
    euler_high = tf.ones([3]) * 0.5
    euler_distribution = tfp.distributions.Uniform(low=euler_low, high=euler_high)

    target_world_frame = trans_distribution.sample(sample_shape=n_samples, seed=aug_opt.seed())
    to_local_frame = tf.reduce_mean(obj_points, axis=1)
    trans_target = target_world_frame - to_local_frame

    euler_target = euler_distribution.sample(sample_shape=n_samples, seed=aug_opt.seed())

    target_params = tf.concat([trans_target, euler_target], 1)

    # pick the most valid transforms, via the learned object state augmentation validity model
    best_target_params = pick_best_params(aug_opt, batch_size, target_params)
    return best_target_params
