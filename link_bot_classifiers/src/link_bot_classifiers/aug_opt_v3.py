from typing import Dict

import tensorflow as tf

import rospy
from link_bot_classifiers.aug_opt_utils import debug_aug, subsample_points, debug_aug_sgd
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from link_bot_pycommon.grid_utils import environment_to_vg_msg, send_voxelgrid_tf_origin_point_res, \
    occupied_voxels_to_points, subtract, binary_or
from merrrt_visualization.rviz_animation_controller import RvizSimpleStepper
from moonshine.geometry import transformation_params_to_matrices, transform_points_3d, pairwise_squared_distances
from moonshine.moonshine_utils import repeat, possibly_none_concat, repeat_tensor


def opt_object_augmentation3(self,
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

    # Sample an initial random object transformation. This can be the same across the batch
    initial_transformation_params = self.scenario.sample_object_augmentation_variables(1000, self.seed)
    # pick the most valid transforms, via the learned object state augmentation validity model
    predicted_errors = self.invariance_model_wrapper.evaluate(initial_transformation_params)
    best_transformation_idx = tf.argmin(predicted_errors)
    best_transformation_params = tf.gather(initial_transformation_params, best_transformation_idx)
    initial_transformation_params = repeat_tensor(best_transformation_params, batch_size, 0, True)

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

                bbox_loss_batch = bbox_loss_v3(self, obj_points_aug)
                bbox_loss = tf.reduce_sum(bbox_loss_batch, axis=-1)

                losses = [
                    tf.reduce_mean(attract_repel_loss_per_point, axis=-1),
                    bbox_loss,
                    invariance_loss,
                ]
                losses_sum = tf.add_n(losses)
                loss = tf.reduce_mean(losses_sum)

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

        can_terminate = self.can_terminate(i, bbox_loss_batch, attract_mask, res, min_dist, gradients)
        if can_terminate:
            break

        clipped_grads_and_vars = self.clip_env_aug_grad(gradients, variables)
        self.opt.apply_gradients(grads_and_vars=clipped_grads_and_vars)

    # this updates other representations of state/action that are fed into the network
    _, object_aug_update, local_origin_point_aug, local_center_aug = self.apply_object_augmentation_no_ik(
        transformation_matrices,
        tf.zeros([batch_size, 1, 3], dtype=tf.float32),
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


def bbox_loss_v3(self, obj_points_aug):
    # FIXME: hardcoded
    lower_extent = tf.constant([-0.6, 0.20, -0.4], tf.float32)
    upper_extent = tf.constant([0.6, 1.2, 1], tf.float32)
    lower_extent_loss = tf.maximum(0, obj_points_aug - upper_extent)
    upper_extent_loss = tf.maximum(0, lower_extent - obj_points_aug)
    bbox_loss = tf.reduce_sum(lower_extent_loss + upper_extent_loss, axis=-1)
    return self.bbox_weight * bbox_loss
