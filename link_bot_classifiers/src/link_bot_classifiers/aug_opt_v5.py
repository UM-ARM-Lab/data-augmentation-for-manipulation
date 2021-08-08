from typing import Dict

import tensorflow as tf

import rospy
import sdf_tools.utils_3d
from link_bot_classifiers.aug_opt_utils import debug_aug, debug_aug_sgd, transformation_obj_points
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from link_bot_pycommon.grid_utils import environment_to_vg_msg, send_voxelgrid_tf_origin_point_res, \
    subtract, binary_or, batch_point_to_idx
from merrrt_visualization.rviz_animation_controller import RvizSimpleStepper
from moonshine.geometry import transformation_params_to_matrices, transformation_jacobian, \
    homogeneous
from moonshine.moonshine_utils import repeat, possibly_none_concat


def opt_object_augmentation5(self,
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
            # stepper.step()

    obj_transforms = opt_object_transform(batch_size, new_env, obj_points, object_points_occupancy, res, self)
    transformation_matrices = transformation_params_to_matrices(obj_transforms, batch_size)
    obj_points_aug, to_local_frame = transformation_obj_points(obj_points, transformation_matrices)

    # this updates other representations of state/action that are fed into the network
    _, object_aug_update, local_origin_point_aug, local_center_aug = self.apply_object_augmentation_no_ik(
        transformation_matrices,
        to_local_frame,
        inputs,
        batch_size,
        time)
    inputs_aug.update(object_aug_update)

    if debug_aug_sgd():
        for b in debug_viz_batch_indices(batch_size):
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

    return inputs_aug, local_origin_point_aug, local_center_aug, local_env_aug_fixed, local_env_aug_fix_deltas


def opt_object_transform(batch_size, new_env, obj_points, object_points_occupancy, res, self):
    initial_transformation_params = self.sample_initial_transforms(batch_size)
    if 'sdf' in new_env and 'sdf_grad' in new_env:
        sdf_no_clipped = new_env['sdf']
        sdf_grad_no_clipped = new_env['sdf_grad']
    else:
        print("Computing SDF online, very slow!")
        sdf_no_clipped, sdf_grad_no_clipped = sdf_tools.utils_3d.compute_sdf_and_gradient(new_env['env'],
                                                                                          new_env['res'],
                                                                                          new_env['origin_point'])
    sdf_no_clipped = tf.convert_to_tensor(sdf_no_clipped)
    sdf_grad_no_clipped = tf.convert_to_tensor(sdf_grad_no_clipped)
    repel_grad_mask = tf.cast(sdf_no_clipped < self.barrier_cut_off, tf.float32)
    repel_sdf_grad = sdf_grad_no_clipped * tf.expand_dims(repel_grad_mask, -1)
    attract_grad_mask = tf.cast(sdf_no_clipped > 0, tf.float32)
    attract_sdf_grad = sdf_grad_no_clipped * tf.expand_dims(attract_grad_mask, -1)
    # optimization loop
    obj_transforms = tf.Variable(initial_transformation_params)  # [x,y,z,roll,pitch,yaw]
    variables = [obj_transforms]
    for i in range(self.max_steps):
        with tf.profiler.experimental.Trace('one_step_loop', i=i):
            with tf.GradientTape() as tape:
                # obj_points is the set of points that define the object state, ie. the swept rope points
                # to compute the object state constraints loss we need to transform this during each forward pass
                # we also need to call apply_object_augmentation* at the end
                # to update the rest of the "state" which is input to the network
                transformation_matrices = transformation_params_to_matrices(obj_transforms, batch_size)
                obj_points_aug, to_local_frame = transformation_obj_points(obj_points, transformation_matrices)

                invariance_loss = self.invariance_weight * self.invariance_model_wrapper.evaluate(obj_transforms)

                bbox_loss_batch = self.bbox_loss(obj_points_aug, new_env['extent'])
                bbox_loss = tf.reduce_sum(bbox_loss_batch, axis=-1)

                losses = [
                    bbox_loss,
                    invariance_loss,
                ]
                losses_sum = tf.add_n(losses)
                loss = tf.reduce_mean(losses_sum)

            gradients = tape.gradient(loss, variables)

            # compute repel and attract gradient via the SDF
            attract_mask = object_points_occupancy  # assumed to already be either 0.0 or 1.0
            obj_point_indices_aug = batch_point_to_idx(obj_points_aug, new_env['res'],
                                                       new_env['origin_point'][None, None])
            oob = tf.logical_or(obj_point_indices_aug < 0,
                                obj_point_indices_aug > tf.convert_to_tensor(new_env['env'].shape, tf.int64)[
                                    None, None])
            oob = tf.reduce_any(oob, axis=-1)
            sdf_dist = tf.gather_nd(sdf_no_clipped, obj_point_indices_aug)  # will be zero if index OOB
            obj_attract_sdf_grad = tf.gather_nd(attract_sdf_grad,
                                                obj_point_indices_aug)  # will be zero if index OOB
            obj_repel_sdf_grad = tf.gather_nd(repel_sdf_grad, obj_point_indices_aug)  # will be zero if index OOB
            attract_grad = obj_attract_sdf_grad * tf.expand_dims(attract_mask, -1) * self.attract_weight
            repel_grad = -obj_repel_sdf_grad * tf.expand_dims((1 - attract_mask), -1) * self.repel_weight
            attract_repel_dpoint = (attract_grad + repel_grad) * self.sdf_grad_scale  # [b,n,3]

            # Compute the jacobian of the transformation
            jacobian = transformation_jacobian(obj_transforms)[:, None]  # [b,1,6,4,4]
            obj_points_local_frame = obj_points - to_local_frame  # [b,n,3]
            obj_points_local_frame_h = homogeneous(obj_points_local_frame)[:, :, None, :, None]  # [b,1,4,1]
            dpoint_dvariables_h = tf.squeeze(tf.matmul(jacobian, obj_points_local_frame_h), axis=-1)  # [b,6]
            dpoint_dvariables = tf.transpose(dpoint_dvariables_h[:, :, :, :3], [0, 1, 3, 2])  # [b,3,6]
            # chain rule
            attract_repel_sdf_grad = tf.einsum('bni,bnij->bnj', attract_repel_dpoint, dpoint_dvariables)  # [b,n,6]
            attract_repel_sdf_grad = tf.reduce_mean(attract_repel_sdf_grad, axis=1)

            # combine with the gradient for the other aspects of the loss, those computed by tf.gradient
            gradients = [gradients[0] + attract_repel_sdf_grad]

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

                self.scenario.plot_points_rviz(attract_points, label='attract', color='g', scale=scale)
                self.scenario.plot_points_rviz(repel_points, label='repel', color='r', scale=scale)
                self.scenario.plot_points_rviz(attract_points_aug, label='attract_aug', color='g', scale=scale)
                self.scenario.plot_points_rviz(repel_points_aug, label='repel_aug', color='r', scale=scale)

                attract_grad_b = -tf.gather(obj_attract_sdf_grad[b], attract_indices, axis=0) * 0.02
                repel_sdf_dist = tf.gather(sdf_dist[b], repel_indices, axis=0)
                repel_oob = tf.gather(oob[b], repel_indices, axis=0)
                repel_sdf_grad_b = tf.gather(obj_repel_sdf_grad[b], repel_indices, axis=0)
                repel_close = tf.logical_and(repel_sdf_dist < self.barrier_cut_off, tf.logical_not(repel_oob))
                repel_close_indices = tf.squeeze(tf.where(repel_close), axis=-1)
                repel_close_grad_b = tf.gather(repel_sdf_grad_b, repel_close_indices, axis=0) * 0.02
                repel_close_points_aug = tf.gather(repel_points_aug, repel_close_indices)
                self.scenario.delete_arrows_rviz(label='attract_sdf_grad')
                self.scenario.delete_arrows_rviz(label='repel_sdf_grad')
                self.scenario.plot_arrows_rviz(attract_points_aug, attract_grad_b, label='attract_sdf_grad',
                                               color='g', scale=0.5)
                self.scenario.plot_arrows_rviz(repel_close_points_aug, repel_close_grad_b, label='repel_sdf_grad',
                                               color='r', scale=0.5)
                # stepper.step()

        can_terminate = self.can_terminate(i, bbox_loss_batch, attract_mask, res, sdf_dist, gradients)
        if can_terminate:
            break

        clipped_grads_and_vars = self.clip_env_aug_grad(gradients, variables)
        self.opt.apply_gradients(grads_and_vars=clipped_grads_and_vars)
    return obj_transforms
