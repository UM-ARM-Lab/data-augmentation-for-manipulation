from dataclasses import dataclass
from typing import Optional, Callable

import tensorflow as tf

from cylinders_simple_demo.aug_opt_utils import transform_obj_points, dpoint_to_dparams, mean_over_moved
from cylinders_simple_demo.grid_utils_tf import batch_point_to_idx, homogeneous
from cylinders_simple_demo.iterative_projection import BaseProjectOpt
from cylinders_simple_demo.utils import empty_callable
# from cylinders_simple_demo.torch_geometry import batch_point_to_idx


@dataclass
class VizVars:
    obj_points_aug: tf.Tensor  # [b, m_objects, n_points, 3]
    to_local_frame: tf.Tensor  # [b, 3]
    min_dist_points_aug: tf.Tensor
    delta_min_dist_grad_dpoint: tf.Tensor
    attract_repel_dpoint: tf.Tensor
    sdf_aug: tf.Tensor


class AugProjOpt(BaseProjectOpt):
    def __init__(self,
                 aug_opt,
                 sdf,
                 sdf_grad,
                 res,
                 origin_point,
                 extent,
                 batch_size,
                 moved_mask,
                 obj_points,
                 obj_occupancy,
                 viz_cb: Callable = empty_callable):
        super().__init__()
        self.aug_opt = aug_opt
        self.sdf = sdf
        self.sdf_grad = sdf_grad
        self.origin_point = origin_point
        self.origin_point_expanded = origin_point[:, None]
        self.origin_point_expanded2 = origin_point[:, None, None]
        self.origin_point_expanded3 = origin_point[:, None, None, None]
        self.res = res
        self.res_expanded = res[:, None]
        self.res_expanded2 = res[:, None, None]
        self.res_expanded3 = res[:, None, None, None]
        self.batch_size = batch_size
        # NOTE: this extent must be in the same frame as the object points
        self.extent = extent
        self.obj_points = obj_points
        self.moved_mask = moved_mask
        self.obj_occupancy = obj_occupancy  # [b,m,T,n_points]
        self.hparams = self.aug_opt.hparams
        self.viz_cb = viz_cb

        # More hyperparameters
        self.step_toward_target_fraction = 1 / self.hparams['n_outer_iters']
        self.lr_decay = 0.90
        self.lr_decay_steps = 10

        # precompute stuff
        obj_point_indices = batch_point_to_idx(self.obj_points, self.res_expanded3, self.origin_point_expanded3)
        obj_sdf = tf.gather_nd(self.sdf, obj_point_indices, batch_dims=1)  # will be zero if index OOB
        # FIXME: we only want to et the object sdf for points where moved_mask is true
        self.obj_sdf_moved = tf.where(tf.cast(moved_mask[..., None, None], tf.bool), obj_sdf, 1e6)
        obj_sdf_flat = tf.reshape(self.obj_sdf_moved, [self.batch_size, -1])
        self.min_dist = tf.reduce_min(obj_sdf_flat, axis=-1)
        self.min_dist_idx = tf.argmin(obj_sdf_flat, axis=-1)  # indexes flattened points

        # viz hyperparameters
        viz_params = self.hparams.get('viz', {})
        self.viz_scale = viz_params.get('scale', 1.0)
        self.viz_arrow_scale = viz_params.get('arrow_scale', 1.0)
        self.viz_delta_min_dist_grad_scale = viz_params.get('delta_min_dist_grad_scale', 4.0)
        self.viz_grad_epsilon = viz_params.get('viz_grad_epsilon', 1e-6)

    def make_opt(self):
        lr = tf.keras.optimizers.schedules.ExponentialDecay(self.hparams['step_size'],
                                                            self.hparams['lr_decay_steps'],
                                                            self.hparams['lr_decay'])
        opt = tf.keras.optimizers.SGD(lr)
        return opt

    def forward(self, tape, obj_transforms):
        s = self.aug_opt.scenario
        with tape:
            # obj_points is the set of points that define the object state, ie. the swept rope points.
            # the obj_points are in world frame, so obj_params is a transform in world frame
            # to compute the object state constraints loss we need to transform this during each forward pass
            # we also need to call apply_object_augmentation* at the end
            # to update the rest of the "state" which is input to the network
            transformation_matrices = s.transformation_params_to_matrices(obj_transforms)
            obj_points_aug, to_local_frame = transform_obj_points(self.obj_points,
                                                                  self.moved_mask,
                                                                  transformation_matrices)

        # compute repel and attract gradient via the SDF
        obj_point_indices_aug = batch_point_to_idx(obj_points_aug, self.res_expanded3, self.origin_point_expanded3)
        obj_sdf_aug = tf.gather_nd(self.sdf, obj_point_indices_aug, batch_dims=1)  # [b, m_objs, T, n_points] 0 if OOB
        obj_sdf_grad_aug = tf.gather_nd(self.sdf_grad, obj_point_indices_aug, batch_dims=1)
        obj_occupancy_aug = tf.cast(obj_sdf_aug < 0, tf.float32)
        obj_occupancy_aug_change = self.obj_occupancy - obj_occupancy_aug
        attract_repel_dpoint = obj_sdf_grad_aug * tf.expand_dims(obj_occupancy_aug_change, -1)
        attract_repel_dpoint = attract_repel_dpoint * self.hparams['sdf_grad_weight']  # [b,n_points,3]

        # and also the grad for preserving the min dist
        obj_sdf_aug_flat = tf.reshape(obj_sdf_aug, [self.batch_size, -1])
        obj_points_aug_flat = tf.reshape(obj_points_aug, [self.batch_size, -1, 3])
        min_dist_aug = tf.gather(obj_sdf_aug_flat, self.min_dist_idx, axis=-1, batch_dims=1)  # [b]
        delta_min_dist = self.min_dist - min_dist_aug
        min_dist_points_aug = tf.gather(obj_points_aug_flat, self.min_dist_idx, axis=-2, batch_dims=1)  # [b,3]
        min_dist_indices_aug = batch_point_to_idx(min_dist_points_aug,
                                                  self.res,
                                                  self.origin_point)  # [b,3]
        delta_min_dist_grad_dpoint = -tf.gather_nd(self.sdf_grad, min_dist_indices_aug, batch_dims=1)  # [b,3]
        delta_min_dist_grad_dpoint = delta_min_dist_grad_dpoint * tf.expand_dims(delta_min_dist, -1)  # [b,3]
        delta_min_dist_grad_dpoint = delta_min_dist_grad_dpoint * self.hparams['delta_min_dist_weight']  # [b,3]

        return VizVars(obj_points_aug=obj_points_aug,
                       to_local_frame=to_local_frame,
                       min_dist_points_aug=min_dist_points_aug,
                       delta_min_dist_grad_dpoint=delta_min_dist_grad_dpoint,
                       attract_repel_dpoint=attract_repel_dpoint,
                       sdf_aug=obj_sdf_aug)

    def project(self, _: int, opt, obj_transforms: tf.Variable):
        s = self.aug_opt.scenario
        tape = tf.GradientTape()

        v = self.forward(tape, obj_transforms)

        with tape:
            invariance_loss = self.aug_opt.invariance_model_wrapper.evaluate(obj_transforms)  # [b, k_transforms]
            # when the constant is larger, this kills the gradient
            invariance_loss = tf.maximum(self.hparams['invariance_threshold'], invariance_loss)
            invariance_loss = self.hparams['invariance_weight'] * invariance_loss
            invariance_loss = tf.reduce_mean(invariance_loss, axis=-1)  # [b]

            bbox_loss_batch = self.aug_opt.bbox_loss(v.obj_points_aug, self.extent)  # [b,k,T,n]
            bbox_loss = tf.reduce_sum(bbox_loss_batch, axis=-1)
            bbox_loss = tf.reduce_sum(bbox_loss, axis=-1)
            bbox_loss = tf.reduce_mean(bbox_loss, axis=-1)  # [b]

            losses = [bbox_loss]
            if not self.aug_opt.no_invariance:
                losses.append(invariance_loss)
            losses_sum = tf.add_n(losses)
            loss = tf.reduce_mean(losses_sum)

        # Compute the jacobian of the transformation. Here the transformation parameters have dimension p
        jacobian = s.aug_transformation_jacobian(obj_transforms)[:, :, None, None]  # [b,k,1,1,p,4,4]
        to_local_frame_moved_mean_expanded = v.to_local_frame[:, None, None, None, :]
        obj_points_local_frame = self.obj_points - to_local_frame_moved_mean_expanded  # [b,m_objects,T,n_points,3]
        obj_points_local_frame_h = homogeneous(obj_points_local_frame)[..., None, :, None]  # [b,m,T,n_points,1,4,1]
        dpoint_dparams_h = tf.squeeze(tf.matmul(jacobian, obj_points_local_frame_h), axis=-1)  # [b,m,T,n_points,p,4]
        dpoint_dparams = dpoint_dparams_h[..., :3]  # [b,m,T,n_points,p,3]
        dpoint_dparams = tf.transpose(dpoint_dparams, [0, 1, 2, 3, 5, 4])  # [b, m, T, n_points, 3, p]

        # chain rule
        attract_repel_sdf_grad = dpoint_to_dparams(v.attract_repel_dpoint, dpoint_dparams)
        attract_repel_sdf_grad = tf.reduce_mean(attract_repel_sdf_grad, -2)
        attract_repel_sdf_grad = tf.reduce_mean(attract_repel_sdf_grad, -2)
        moved_attract_repel_sdf_grad_mean = mean_over_moved(self.moved_mask, attract_repel_sdf_grad)  # [b, p]
        # NOTE: this code isn't general enough to handle multiple transformations (k>1)
        moved_attract_repel_sdf_grad_mean = tf.expand_dims(moved_attract_repel_sdf_grad_mean, axis=-2)

        p = obj_transforms.shape[-1]
        dpoint_dparams_flat = tf.reshape(dpoint_dparams, [self.batch_size, -1, 3, p])
        dpoint_dparams_for_min_point = tf.gather(dpoint_dparams_flat, self.min_dist_idx, batch_dims=1)  # [b,p]
        delta_min_dist_grad_dparams = dpoint_to_dparams(v.delta_min_dist_grad_dpoint,  # [b, 3],
                                                        dpoint_dparams_for_min_point)  # [b, 3, p] -> [b, p]
        delta_min_dist_grad_dparams = tf.expand_dims(delta_min_dist_grad_dparams, axis=-2)

        variables = [obj_transforms]
        tape_gradients = tape.gradient(loss, variables)

        # combine with the gradient for the other aspects of the loss, those computed by tf.gradient
        gradients = tape_gradients[0]
        if not self.aug_opt.no_occupancy:
            gradients += moved_attract_repel_sdf_grad_mean
        if not self.aug_opt.no_delta_min_dist:
            gradients += delta_min_dist_grad_dparams

        clipped_grads_and_vars = self.clip_env_aug_grad([gradients], variables)
        opt.apply_gradients(grads_and_vars=clipped_grads_and_vars)
        lr = opt._decayed_lr(tf.float32)
        can_terminate = self.can_terminate(lr, gradients)

        x_out = tf.identity(obj_transforms)
        return x_out, can_terminate, v

    def clip_env_aug_grad(self, gradients, variables):
        def _clip(g):
            # we want grad_clip to be as close to in meters as possible, so here we scale by step size
            c = self.hparams['grad_clip'] / self.hparams['step_size']
            return tf.clip_by_value(g, -c, c)

        return [(_clip(g), v) for (g, v) in zip(gradients, variables)]

    def can_terminate(self, lr, gradients):
        grad_norm = tf.linalg.norm(gradients[0], axis=-1)
        step_size_i = grad_norm * lr
        can_terminate = step_size_i < self.hparams['step_size_threshold']
        all_can_terminate = tf.reduce_all(can_terminate)
        return all_can_terminate

    def step_towards_target(self, target_transforms, obj_transforms):
        # NOTE: although interpolating euler angles can be problematic or unintuitive,
        #  we have ensured the differences are <pi/2. So it should be ok
        x_interp = obj_transforms + (target_transforms - obj_transforms) * self.step_toward_target_fraction
        tape = tf.GradientTape()
        viz_vars = self.forward(tape, x_interp)
        return x_interp, viz_vars

    def viz_func(self, _: Optional, obj_transforms, __, target, v: Optional[VizVars]):
        pass
