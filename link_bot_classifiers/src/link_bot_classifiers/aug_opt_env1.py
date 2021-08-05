from typing import Dict

import tensorflow as tf

import rospy
from link_bot_classifiers.aug_opt_utils import debug_aug, subsample_points, debug_aug_sgd, EnvPoints, MinDists, \
    EnvOptDebugVars
from link_bot_data.dataset_utils import add_predicted
from link_bot_pycommon.bbox_visualization import grid_to_bbox
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from link_bot_pycommon.grid_utils import environment_to_vg_msg, send_voxelgrid_tf_origin_point_res, \
    occupied_voxels_to_points, subtract, binary_or
from merrrt_visualization.rviz_animation_controller import RvizSimpleStepper
from moonshine.geometry import pairwise_squared_distances
from moonshine.moonshine_utils import reduce_mean_no_nan


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
        for b in debug_viz_batch_indices(batch_size):
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
                        repel_close_indices = tf.squeeze(tf.where(min_dists.repel < self.barrier_cut_off),
                                                         axis=-1)
                        robot_repel_close_indices = tf.squeeze(
                            tf.where(min_dists.robot_repel < self.barrier_cut_off),
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
