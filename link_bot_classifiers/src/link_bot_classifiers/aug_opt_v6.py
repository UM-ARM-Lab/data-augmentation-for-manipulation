from typing import Dict

import tensorflow as tf
import tensorflow_probability as tfp

import rospy
from link_bot_classifiers.aug_opt_utils import debug_aug, debug_aug_sgd, transformation_obj_points
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

n_outer_iters = 10
delta_min_dist_threshold = 0.055
not_progressing_threshold = 0.001
loss_threshold = 0.001
max_env_violations = 8


class AugV6ProjOpt(BaseProjectOpt):
    def __init__(self, aug_opt, new_env, res, batch_size, obj_points, object_points_occupancy):
        super().__init__()
        self.aug_opt = aug_opt
        self.new_env = new_env
        self.res = res
        self.batch_size = batch_size
        self.obj_points = obj_points
        self.object_points_occupancy = object_points_occupancy

        # More hyperparameters
        self.step_toward_target_fraction = 1 / n_outer_iters
        self.lr_decay = 0.90
        self.lr_decay_steps = 10

        self.aug_dir_pub = rospy.Publisher('aug_dir', Marker, queue_size=10)

        if 'sdf' in self.new_env and 'sdf_grad' in self.new_env:
            sdf_no_clipped = self.new_env['sdf']
            sdf_grad_no_clipped = self.new_env['sdf_grad']
        else:
            print("Computing SDF online, very slow!")
            sdf_no_clipped, sdf_grad_no_clipped = utils_3d.compute_sdf_and_gradient(self.new_env['env'],
                                                                                    self.new_env['res'],
                                                                                    self.new_env['origin_point'])

        self.sdf_no_clipped = tf.convert_to_tensor(sdf_no_clipped)
        sdf_grad_no_clipped = tf.convert_to_tensor(sdf_grad_no_clipped)
        repel_grad_mask = tf.cast(sdf_no_clipped < self.aug_opt.barrier_cut_off, tf.float32)
        self.repel_sdf_grad = sdf_grad_no_clipped * tf.expand_dims(repel_grad_mask, -1)
        attract_grad_mask = tf.cast(sdf_no_clipped > 0, tf.float32)
        self.attract_sdf_grad = sdf_grad_no_clipped * tf.expand_dims(attract_grad_mask, -1)

    def make_opt(self):
        lr = tf.keras.optimizers.schedules.ExponentialDecay(self.aug_opt.step_size, self.lr_decay_steps, self.lr_decay)
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
        obj_point_indices_aug = batch_point_to_idx(obj_points_aug, self.new_env['res'],
                                                   self.new_env['origin_point'][None, None])
        sdf_dist = tf.gather_nd(self.sdf_no_clipped, obj_point_indices_aug)  # will be zero if index OOB
        obj_attract_sdf_grad = tf.gather_nd(self.attract_sdf_grad,
                                            obj_point_indices_aug)  # will be zero if index OOB
        obj_repel_sdf_grad = tf.gather_nd(self.repel_sdf_grad, obj_point_indices_aug)  # will be zero if index OOB

        return obj_attract_sdf_grad, obj_repel_sdf_grad, sdf_dist, obj_point_indices_aug, obj_points_aug, to_local_frame

    def step(self, i: int, opt, obj_transforms: tf.Variable):
        tape = tf.GradientTape()

        viz_vars = self.forward(tape, obj_transforms)
        obj_attract_sdf_grad, obj_repel_sdf_grad, sdf_dist, obj_point_indices_aug, obj_points_aug, to_local_frame = viz_vars

        with tape:
            invariance_loss = self.aug_opt.invariance_weight * self.aug_opt.invariance_model_wrapper.evaluate(
                obj_transforms)

            bbox_loss_batch = self.aug_opt.bbox_loss(obj_points_aug, self.new_env['extent'])
            bbox_loss = tf.reduce_sum(bbox_loss_batch, axis=-1)

            losses = [
                bbox_loss,
                invariance_loss,
            ]
            losses_sum = tf.add_n(losses)
            loss = tf.reduce_mean(losses_sum)

        attract_mask = self.object_points_occupancy  # assumed to already be either 0.0 or 1.0
        attract_grad = obj_attract_sdf_grad * tf.expand_dims(attract_mask, -1) * self.aug_opt.attract_weight
        repel_grad = -obj_repel_sdf_grad * tf.expand_dims((1 - attract_mask), -1) * self.aug_opt.repel_weight
        attract_repel_dpoint = (attract_grad + repel_grad)  # [b,n,3]

        # Compute the jacobian of the transformation
        jacobian = transformation_jacobian(obj_transforms)[:, None]  # [b,1,6,4,4]
        obj_points_local_frame = self.obj_points - to_local_frame  # [b,n,3]
        obj_points_local_frame_h = homogeneous(obj_points_local_frame)[:, :, None, :, None]  # [b,1,4,1]
        dpoint_dvariables_h = tf.squeeze(tf.matmul(jacobian, obj_points_local_frame_h), axis=-1)  # [b,6]
        dpoint_dvariables = tf.transpose(dpoint_dvariables_h[:, :, :, :3], [0, 1, 3, 2])  # [b,3,6]

        # chain rule
        attract_repel_sdf_grad = tf.einsum('bni,bnij->bnj', attract_repel_dpoint, dpoint_dvariables)  # [b,n,6]
        attract_repel_sdf_grad = tf.reduce_mean(attract_repel_sdf_grad, axis=1)

        variables = [obj_transforms]
        gradients = tape.gradient(loss, variables)

        # combine with the gradient for the other aspects of the loss, those computed by tf.gradient
        gradients = [gradients[0] + attract_repel_sdf_grad]

        clipped_grads_and_vars = self.aug_opt.clip_env_aug_grad(gradients, variables)
        opt.apply_gradients(grads_and_vars=clipped_grads_and_vars)
        can_terminate = self.aug_opt.can_terminate(i, bbox_loss_batch, attract_mask, self.res, sdf_dist, gradients)

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

    def viz_func(self, i, obj_transforms, initial_value, target, viz_vars):
        obj_attract_sdf_grad, obj_repel_sdf_grad, sdf_dist, obj_point_indices_aug, obj_points_aug, to_local_frame = viz_vars

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
                repel_sdf_dist = tf.gather(sdf_dist[b], repel_indices, axis=0)
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

    obj_point_indices = batch_point_to_idx(obj_points, new_env['res'], new_env['origin_point'][None, None])
    sdf_dist = tf.gather_nd(new_env['sdf'], obj_point_indices)  # will be zero if index OOB

    initial_transformation_params = initial_identity_params(batch_size)
    target_transformation_params = sample_target_transform_params(obj_points, new_env, batch_size, aug_opt.seed)
    project_opt = AugV6ProjOpt(aug_opt=aug_opt, new_env=new_env, res=res, batch_size=batch_size, obj_points=obj_points,
                               object_points_occupancy=object_points_occupancy)
    obj_transforms, viz_vars = iterative_projection(initial_value=initial_transformation_params,
                                                    target=target_transformation_params,
                                                    n=n_outer_iters,
                                                    m=aug_opt.max_steps,
                                                    step_towards_target=project_opt.step_towards_target,
                                                    project_opt=project_opt,
                                                    x_distance=project_opt.distance,
                                                    not_progressing_threshold=not_progressing_threshold,
                                                    viz_func=project_opt.viz_func)
    _, __, sdf_dist_aug, ___, ____, ______ = viz_vars

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
    squared_res_expanded = tf.square(res)[:, None]
    attract_satisfied = tf.cast(sdf_dist_aug < squared_res_expanded, tf.float32)
    repel_satisfied = tf.cast(sdf_dist_aug > squared_res_expanded, tf.float32)
    env_constraints_satisfied_ = (attract_mask * attract_satisfied) + ((1 - attract_mask) * repel_satisfied)
    num_env_constraints_violated = tf.reduce_sum(1 - env_constraints_satisfied_, axis=1)
    env_constraints_satisfied = tf.cast(num_env_constraints_violated < max_env_violations, tf.float32)

    min_dist = tf.reduce_min(sdf_dist, axis=1)
    min_dist_aug = tf.reduce_min(sdf_dist_aug, axis=1)
    delta_min_dist = tf.abs(min_dist - min_dist_aug)
    delta_min_dist_satisfied = tf.cast(delta_min_dist < delta_min_dist_threshold, tf.float32)

    constraints_satisfied = env_constraints_satisfied * bbox_constraint_satisfied * delta_min_dist_satisfied
    return constraints_satisfied


def sample_target_transform_params(obj_points, new_env, batch_size, seed: tfp.util.SeedStream):
    to_local_frame = tf.reduce_mean(obj_points, axis=1)
    extent = tf.reshape(new_env['extent'], [3, 2])
    low = extent[None, :, 0]
    high = extent[None, :, 1]
    distribution = tfp.distributions.Uniform(low=low, high=high)
    target_world_frame = distribution.sample(seed=seed())
    trans_target = target_world_frame - to_local_frame
    target_params = tf.concat([trans_target, tf.zeros([batch_size, 3])], 1)
    return target_params


def initial_identity_params(batch_size):
    return tf.zeros([batch_size, 6], tf.float32)
