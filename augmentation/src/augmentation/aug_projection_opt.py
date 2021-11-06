from dataclasses import dataclass
from typing import Optional

import tensorflow as tf
from matplotlib import cm

import rospy
from augmentation.aug_opt_utils import transform_obj_points, dpoint_to_dparams
from augmentation.iterative_projection import BaseProjectOpt
from link_bot_data.rviz_arrow import rviz_arrow
from link_bot_data.visualization_common import make_delete_marker
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from link_bot_pycommon.grid_utils import batch_point_to_idx
from moonshine.geometry import transformation_params_to_matrices, transformation_jacobian, homogeneous, euler_angle_diff
from tf import transformations
from visualization_msgs.msg import Marker


@dataclass
class VizVars:
    obj_points_aug: tf.Tensor  # [b, m, n_points, 3]
    to_local_frame: tf.Tensor  # [b, m, 1, 3]
    min_dist_points_aug: tf.Tensor
    delta_min_dist_grad_dpoint: tf.Tensor
    attract_repel_dpoint: tf.Tensor
    sdf_aug: tf.Tensor


class AugProjOpt(BaseProjectOpt):
    def __init__(self, aug_opt, sdf, sdf_grad, res, origin_point, extent, batch_size, obj_points, obj_occupancy):
        super().__init__()
        self.aug_opt = aug_opt
        self.sdf = sdf
        self.sdf_grad = sdf_grad
        self.origin_point = origin_point
        self.origin_point_expanded = origin_point[:, None]
        self.origin_point_expanded2 = origin_point[:, None, None]
        self.res = res
        self.res_expanded = res[:, None]
        self.res_expanded2 = res[:, None, None]
        self.batch_size = batch_size
        self.extent = extent
        self.obj_points = obj_points
        self.obj_occupancy = obj_occupancy  # [b,m,n_points]
        self.hparams = self.aug_opt.hparams

        # More hyperparameters
        self.step_toward_target_fraction = 1 / self.hparams['n_outer_iters']
        self.lr_decay = 0.90
        self.lr_decay_steps = 10

        self.aug_dir_pub = rospy.Publisher('aug_dir', Marker, queue_size=10)

        # precompute stuff
        self.obj_point_indices = batch_point_to_idx(self.obj_points, self.res_expanded2, self.origin_point_expanded2)
        self.obj_sdf = tf.gather_nd(self.sdf, self.obj_point_indices, batch_dims=1)  # will be zero if index OOB
        self.min_dist = tf.reduce_min(self.obj_sdf, axis=-1)
        self.min_dist_idx = tf.argmin(self.obj_sdf, axis=-1)

        # viz hyperparameters
        viz_params = self.hparams.get('viz', {})
        self.viz_scale = viz_params.get('scale', 1.0)
        self.viz_arrow_scale = viz_params.get('arrow_scale', 1.0)
        self.viz_min_delta_dist_grad_scale = viz_params.get('min_delta_dist_grad_scale', 4.0)

    def make_opt(self):
        lr = tf.keras.optimizers.schedules.ExponentialDecay(self.hparams['step_size'],
                                                            self.hparams['lr_decay_steps'],
                                                            self.hparams['lr_decay'])
        opt = tf.keras.optimizers.SGD(lr)
        return opt

    def forward(self, tape, obj_transforms):
        with tape:
            # obj_points is the set of points that define the object state, ie. the swept rope points.
            # the obj_points are in world frame, so obj_params is a transform in world frame
            # to compute the object state constraints loss we need to transform this during each forward pass
            # we also need to call apply_object_augmentation* at the end
            # to update the rest of the "state" which is input to the network
            transformation_matrices = transformation_params_to_matrices(obj_transforms)
            obj_points_aug, to_local_frame = transform_obj_points(self.obj_points, transformation_matrices)

        # compute repel and attract gradient via the SDF
        obj_point_indices_aug = batch_point_to_idx(obj_points_aug, self.res_expanded2, self.origin_point_expanded2)
        obj_sdf_aug = tf.gather_nd(self.sdf, obj_point_indices_aug, batch_dims=1)  # will be zero if index OOB
        obj_sdf_grad_aug = tf.gather_nd(self.sdf_grad, obj_point_indices_aug, batch_dims=1)
        obj_occupancy_aug = tf.cast(obj_sdf_aug < 0, tf.float32)
        obj_occupancy_aug_change = self.obj_occupancy - obj_occupancy_aug
        attract_repel_dpoint = obj_sdf_grad_aug * tf.expand_dims(obj_occupancy_aug_change, -1)
        attract_repel_dpoint = attract_repel_dpoint * self.hparams['sdf_grad_weight']  # [b,m,n_points,3]

        # and also the grad for preserving the min dist
        min_dist_aug = tf.gather(obj_sdf_aug, self.min_dist_idx, axis=-1, batch_dims=2)  # [b,m]
        delta_min_dist = self.min_dist - min_dist_aug
        min_dist_points_aug = tf.gather(obj_points_aug, self.min_dist_idx, axis=-2, batch_dims=2)  # [b,m,3]
        min_dist_indices_aug = batch_point_to_idx(min_dist_points_aug,
                                                  self.res_expanded,
                                                  self.origin_point_expanded)  # [b,m,3]
        delta_min_dist_grad_dpoint = -tf.gather_nd(self.sdf_grad, min_dist_indices_aug, batch_dims=1)  # [b,m,3]
        delta_min_dist_grad_dpoint = delta_min_dist_grad_dpoint * tf.expand_dims(delta_min_dist, -1)  # [b,m,3]
        delta_min_dist_grad_dpoint = delta_min_dist_grad_dpoint * self.hparams['delta_min_dist_weight']  # [b,m,3]

        return VizVars(obj_points_aug=obj_points_aug,
                       to_local_frame=to_local_frame,
                       min_dist_points_aug=min_dist_points_aug,
                       delta_min_dist_grad_dpoint=delta_min_dist_grad_dpoint,
                       attract_repel_dpoint=attract_repel_dpoint,
                       sdf_aug=obj_sdf_aug)

    def step(self, _: int, opt, obj_transforms: tf.Variable):
        tape = tf.GradientTape()

        v = self.forward(tape, obj_transforms)

        with tape:
            invariance_loss = self.aug_opt.invariance_model_wrapper.evaluate(obj_transforms)
            invariance_loss = tf.maximum(self.hparams['invariance_threshold'], invariance_loss)
            invariance_loss = self.hparams['invariance_weight'] * invariance_loss
            invariance_loss = tf.reduce_mean(invariance_loss, axis=-1)

            bbox_loss_batch = self.aug_opt.bbox_loss(v.obj_points_aug, self.extent)  # [b,m]
            bbox_loss = tf.reduce_sum(bbox_loss_batch, axis=-1)
            bbox_loss = tf.reduce_mean(bbox_loss, axis=-1)  # [b]

            losses = [
                bbox_loss,
                invariance_loss,
            ]
            losses_sum = tf.add_n(losses)
            loss = tf.reduce_mean(losses_sum)

        # Compute the jacobian of the transformation
        jacobian = tf.expand_dims(transformation_jacobian(obj_transforms), axis=-4)  # [b,m,1,6,4,4]
        obj_points_local_frame = self.obj_points - v.to_local_frame  # [b,m,n_points,3]
        obj_points_local_frame_h = homogeneous(obj_points_local_frame)[..., None, :, None]  # [b,m,n_points,1,4,1]
        dpoint_dparams_h = tf.squeeze(tf.matmul(jacobian, obj_points_local_frame_h), axis=-1)  # [b,m,n_points,6,4]
        dpoint_dparams = dpoint_dparams_h[..., :3]  # [b,m,n_points,6,3]
        dpoint_dparams = tf.transpose(dpoint_dparams, [0, 1, 2, 4, 3])  # [b,m,n_points,3,6]

        # chain rule
        attract_repel_sdf_grad = dpoint_to_dparams(v.attract_repel_dpoint, dpoint_dparams)
        dpoint_dparams_for_min_point = tf.gather(dpoint_dparams, self.min_dist_idx, axis=-3, batch_dims=2)  # [b,m,3,6]
        delta_min_dist_grad_dparams = tf.einsum('bmi,bmij->bmj', v.delta_min_dist_grad_dpoint,
                                                dpoint_dparams_for_min_point)  # [b,m,6]

        variables = [obj_transforms]
        tape_gradients = tape.gradient(loss, variables)

        # combine with the gradient for the other aspects of the loss, those computed by tf.gradient
        gradients = [tape_gradients[0] + attract_repel_sdf_grad + delta_min_dist_grad_dparams]

        clipped_grads_and_vars = self.clip_env_aug_grad(gradients, variables)
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
        can_terminate = tf.reduce_all(can_terminate)
        return can_terminate

    def step_towards_target(self, target_transforms, obj_transforms):
        # NOTE: although interpolating euler angles can be problematic or unintuitive,
        #  we have ensured the differences are <pi/2. So it should be ok
        x_interp = obj_transforms + (target_transforms - obj_transforms) * self.step_toward_target_fraction
        tape = tf.GradientTape()
        viz_vars = self.forward(tape, x_interp)
        return x_interp, viz_vars

    def distance(self, transforms1, transforms2):
        trans1 = transforms1[..., :3]
        trans2 = transforms2[..., :3]
        euler1 = transforms1[..., 3:]
        euler2 = transforms2[..., 3:]
        euler_dist = tf.linalg.norm(euler_angle_diff(euler1, euler2), axis=-1)
        trans_dist = tf.linalg.norm(trans1 - trans2, axis=-1)
        distances = trans_dist + euler_dist
        max_distance = tf.reduce_max(distances)
        return max_distance

    def viz_func(self, _: Optional, obj_transforms, __, target, v: Optional[VizVars]):
        for b in debug_viz_batch_indices(self.batch_size):
            target_b = target[b]
            target_pos_b = target_b[..., :3].numpy()

            obj_transforms_b = obj_transforms[b]

            for obj_i in range(target_b.shape[0]):
                obj_transforms_b_i = obj_transforms_b[obj_i]
                target_pos_b_i = target_pos_b[obj_i]
                target_b_i = target_b[obj_i]

                self.plot_transform(obj_i, obj_transforms_b_i, f'aug_opt_current_{obj_i}')
                self.plot_transform(obj_i, target_b_i, f'aug_opt_target_{obj_i}')

                dir_msg = rviz_arrow([0, 0, 0], target_pos_b_i, scale=2.0)
                dir_msg.header.frame_id = f'aug_opt_initial_{obj_i}'
                dir_msg.id = obj_i
                self.aug_dir_pub.publish(dir_msg)

                obj_points_b_i = self.obj_points[b, obj_i]  # [n_points, 3]
                obj_occupancy_b_i = self.obj_occupancy[b, obj_i]  # [n_points]

                repel_indices = tf.squeeze(tf.where(1 - obj_occupancy_b_i))  # [n_repel_points]
                attract_indices = tf.squeeze(tf.where(obj_occupancy_b_i), -1)  # [n_attract_points]

                self.viz_b_i(attract_indices,
                             b,
                             obj_i,
                             obj_points_b_i,
                             obj_transforms_b_i,
                             repel_indices,
                             target_b_i,
                             target_pos_b_i,
                             v)

    def viz_b_i(self,
                attract_indices,
                b,
                obj_i,
                obj_points_b_i,
                obj_transforms_b_i,
                repel_indices,
                target_b_i,
                target_pos_b_i,
                v):
        s = self.aug_opt.scenario
        s.plot_aug_points_rviz(obj_i, obj_points_b_i, '', cm.Greys)
        repel_points = tf.gather(obj_points_b_i, repel_indices).numpy()  # [n_attract_points]
        attract_points = tf.gather(obj_points_b_i, attract_indices).numpy()  # [n_repel_points]
        s.plot_points_rviz(attract_points, label=f'attract_{obj_i}', color='g', scale=self.viz_scale)
        s.plot_points_rviz(repel_points, label=f'repel_{obj_i}', color='r', scale=self.viz_scale)
        if v is not None:
            local_pos_b = v.to_local_frame[b, obj_i, 0].numpy()  # [3]
            obj_points_aug_b_i = v.obj_points_aug[b, obj_i]  # [n_points, 3]
            s.plot_aug_points_rviz(obj_i, obj_points_aug_b_i, 'aug', cm.Blues)
            self.aug_opt.debug.send_position_transform(local_pos_b, f'aug_opt_initial_{obj_i}')
            attract_points_aug = tf.gather(obj_points_aug_b_i, attract_indices).numpy()
            repel_points_aug = tf.gather(obj_points_aug_b_i, repel_indices).numpy()
            s.plot_points_rviz(attract_points_aug, f'attract_aug_{obj_i}', color='g', scale=self.viz_scale)
            s.plot_points_rviz(repel_points_aug, f'repel_aug_{obj_i}', color='r', scale=self.viz_scale)

            attract_repel_dpoint_b_i = v.attract_repel_dpoint[b, obj_i]
            attract_grad_b = -tf.gather(attract_repel_dpoint_b_i, attract_indices, axis=0) * 0.02
            repel_grad_b = -tf.gather(attract_repel_dpoint_b_i, repel_indices, axis=0) * 0.02
            s.plot_arrows_rviz(attract_points_aug, attract_grad_b, f'attract_sdf_grad_{obj_i}', color='g',
                               scale=self.viz_arrow_scale)
            s.plot_arrows_rviz(repel_points_aug, repel_grad_b, f'repel_sdf_grad_{obj_i}', color='r',
                               scale=self.viz_arrow_scale)

            delta_min_dist_points_b_i = v.min_dist_points_aug[b, obj_i].numpy()
            delta_min_dist_grad_b_i = -v.delta_min_dist_grad_dpoint[b, obj_i].numpy()
            s.plot_arrow_rviz(delta_min_dist_points_b_i,
                              delta_min_dist_grad_b_i * self.viz_min_delta_dist_grad_scale,
                              label=f'delta_min_dist_grad_{obj_i}',
                              color='pink',
                              scale=self.viz_arrow_scale)

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
        self.aug_opt.scenario.tf.send_transform(target_pos_b, target_q_b, f'aug_opt_initial_{obj_i}', frame_id, False)

    def clear_viz(self):
        s = self.aug_opt.scenario
        for _ in range(3):
            self.aug_dir_pub.publish(make_delete_marker(ns='arrow'))
            s.point_pub.publish(make_delete_marker(ns='attract'))
            s.point_pub.publish(make_delete_marker(ns='repel'))
            s.point_pub.publish(make_delete_marker(ns='attract_aug'))
            s.point_pub.publish(make_delete_marker(ns='repel_aug'))
            s.delete_arrows_rviz(label='delta_min_dist_grad')
            s.delete_arrows_rviz(label='attract_sdf_grad')
            s.delete_arrows_rviz(label='repel_sdf_grad')
            rospy.sleep(0.01)