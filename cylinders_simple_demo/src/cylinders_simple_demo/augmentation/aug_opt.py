from typing import Dict, List, Callable

import numpy as np
import torch

from cylinders_simple_demo.augmentation.aug_opt_utils import pick_best_params, transform_obj_points, \
    sum_over_moved, check_env_constraints
from cylinders_simple_demo.augmentation.aug_projection_opt import AugProjOpt
from cylinders_simple_demo.augmentation.iterative_projection import iterative_projection
from cylinders_simple_demo.utils.cylinders_scenario import CylindersScenario
from cylinders_simple_demo.utils.torch_geometry import lookup_points_in_vg, \
    points_to_voxel_grid_res_origin_point_batched
from cylinders_simple_demo.utils.utils import has_keys

cache_ = {}


class AcceptInvarianceModel:
    def __init__(self):
        pass

    def evaluate(self, sampled_params):
        return torch.zeros(sampled_params.shape[:-1], device=sampled_params.device)


def compute_moved_mask(obj_points, moved_threshold=0.01):
    """
    obj_points: [b, m, T, n_points, 3]
    return: [b, m], [b, m]
    """
    obj_points_dist = (obj_points - obj_points[:, :, 0:1]).norm(dim=-1)  # [b, m, T, n_points]
    obj_points_dist = torch.max(torch.max(obj_points_dist, dim=-1)[0], dim=-1)[0]  # [b, m]
    moved_mask = obj_points_dist > moved_threshold ** 2
    robot_always_moved_mask = torch.zeros_like(moved_mask)
    robot_always_moved_mask[:, 0] = 1
    moved_mask = torch.logical_or(moved_mask, robot_always_moved_mask)
    return moved_mask.float()


def add_stationary_points_to_env(env, obj_points, moved_mask, res, origin_point, batch_size, device):
    """

    Args:
        env: [b, h, w, c]
        obj_points:  [b, m_objects, T, n_points, 3]
        moved_mask:  [b, m_objects]
        res:
        origin_point:
        batch_size:

    Returns:

    """
    batch_indices, moved_object_indices = torch.where(1 - moved_mask)
    points = obj_points[batch_indices, moved_object_indices]  # [n, T, n_points, 3]

    # because the objects here are stationary, we can ignore the time dimension
    points_0 = points[:, 0]  # [b, n_points, 3]
    n_points = points_0.shape[1]

    points_flat = points_0.reshape([-1, 3])
    res_flat = res[batch_indices].repeat_interleave(n_points)
    origin_point_flat = origin_point[batch_indices].repeat_interleave(n_points, 0)
    batch_indices_flat = batch_indices.repeat_interleave(n_points)

    # [b, h, w, c]
    env_stationary = points_to_voxel_grid_res_origin_point_batched(batch_indices_flat,
                                                                   points_flat,
                                                                   res_flat,
                                                                   origin_point_flat,
                                                                   *env.shape[-3:],
                                                                   batch_size,
                                                                   device)
    env_stationary = torch.clip(env + env_stationary, 0, 1)
    return env_stationary


class AugmentationOptimization:

    def __init__(self, scenario: CylindersScenario, hparams: Dict, batch_size: int, state_keys: List[str],
                 action_keys: List[str]):
        self.state_keys = state_keys
        self.action_keys = action_keys
        self.hparams = hparams.get('augmentation', None)
        self.batch_size = batch_size
        self.scenario = scenario

        self.seed = 4 if self.hparams is None or 'seed' not in self.hparams else self.hparams['seed']
        self.rng = np.random.RandomState(self.seed)

        if self.do_augmentation():
            self.invariance_model_wrapper = AcceptInvarianceModel()

            # ablations
            self.no_invariance = has_keys(self.hparams, ['ablations', 'no_invariance'], False)
            self.no_occupancy = has_keys(self.hparams, ['ablations', 'no_occupancy'], False)
            self.no_delta_min_dist = has_keys(self.hparams, ['ablations', 'no_delta_min_dist'], False)

    def aug_opt(self, inputs: Dict, batch_size: int, time: int, device):
        res = inputs['res']
        extent = inputs['extent']
        origin_point = inputs['origin_point']
        env = inputs['env']

        # to avoid confusion, just remove these from inputs
        if 'sdf' in inputs:
            inputs.pop("sdf")
        if 'sdf_grad' in inputs:
            inputs.pop("sdf_grad")

        obj_points = self.scenario.compute_obj_points(inputs, batch_size, device)  # [b,m,T,num_points,3]
        # check which objects move over time
        moved_mask = compute_moved_mask(obj_points)  # [b, m_objects]
        obj_points_flat = torch.reshape(obj_points, [batch_size, -1, 3])
        obj_occupancy_flat = lookup_points_in_vg(obj_points_flat, env, res, origin_point, batch_size)
        obj_occupancy = torch.reshape(obj_occupancy_flat, obj_points.shape[:-1])  # [b, m, num_points]

        # then we can add the points that represent the non-moved objects to the "env" voxel grid,
        # then compute SDF and grad. This will be slow, what can we do about that?
        # get all components of the state as a set of points. this could be the swept volume and/or include the robot
        env_stationary = add_stationary_points_to_env(env,
                                                      obj_points,
                                                      moved_mask,
                                                      res,
                                                      origin_point,
                                                      batch_size,
                                                      device)

        # TODO: replace with pure pytorch implementation
        from moonshine.tfa_sdf import compute_sdf_and_gradient_batch
        sdf_stationary, sdf_grad_stationary = compute_sdf_and_gradient_batch(env_stationary.cpu(), res.cpu())
        sdf_stationary = torch.from_numpy(sdf_stationary.numpy()).to(device)
        sdf_grad_stationary = torch.from_numpy(sdf_grad_stationary.numpy()).to(device)

        def _viz_cb(_b):
            pass

        transformation_matrices, to_local_frame, is_obj_aug_valid = self.aug_obj_transform(
            res=res,
            extent=extent,
            origin_point=origin_point,
            sdf=sdf_stationary,
            sdf_grad=sdf_grad_stationary,
            moved_mask=moved_mask,
            obj_points=obj_points,
            obj_occupancy=obj_occupancy,
            viz_cb=_viz_cb,
            batch_size=batch_size,
            device=device)

        # apply the transformations to some components of the state/action
        obj_aug_update, local_origin_point_aug, local_center_aug = self.aug_apply_no_ik(
            moved_mask=moved_mask,
            transformation_matrices=transformation_matrices,
            to_local_frame=to_local_frame,
            inputs=inputs,
            batch_size=batch_size,
            device=device)

        keys_aug = list(obj_aug_update.keys())

        # things that we won't be updating in this augmentation
        inputs_aug = {
            'batch_size':   batch_size,
            'env':          env,
            'extent':       extent,
            'origin_point': inputs['origin_point'],
            'res':          res,
            'sdf':          sdf_stationary,
            'sdf_grad':     sdf_grad_stationary,
        }
        inputs_aug.update(obj_aug_update)
        inputs_aug.update(self.scenario.aug_copy_inputs(inputs))

        # NOTE: We use IK as a simple and efficient way to preserve the contacts between the robot and the environment.
        #  Preserving contacts is a key insight of our augmentation method, so in a way this is just a more specific
        #  implementation of a more abstract rule. Solving IK is very efficient, but a bit less general.
        #  it assumes the body of the robot is not in contact and that any valid position IK is similar enough
        is_ik_valid, ik_keys_aug = self.scenario.aug_ik(inputs=inputs,
                                                        inputs_aug=inputs_aug,
                                                        batch_size=batch_size)
        keys_aug += ik_keys_aug

        is_valid = is_ik_valid * is_obj_aug_valid
        print(is_valid.sum() / is_valid.numel())

        inputs_aug = self.use_original_if_invalid(is_valid, batch_size, inputs, inputs_aug, keys_aug)

        # add some more useful info
        inputs_aug['is_valid'] = is_valid

        return inputs_aug

    def aug_obj_transform(self,
                          res,
                          extent,
                          origin_point,
                          sdf,
                          sdf_grad,
                          moved_mask,
                          obj_points,
                          obj_occupancy,
                          viz_cb: Callable,
                          batch_size: int,
                          device,
                          ):
        k_transforms = 1  # this is always one at the moment because we transform all moved objects rigidly
        initial_transformation_params = self.scenario.initial_identity_aug_params(batch_size, k_transforms, device)
        target_transformation_params = self.sample_target_transform_params(batch_size, k_transforms, device)
        project_opt = AugProjOpt(aug_opt=self,
                                 sdf=sdf,
                                 sdf_grad=sdf_grad,
                                 res=res,
                                 origin_point=origin_point,
                                 extent=extent,
                                 batch_size=batch_size,
                                 moved_mask=moved_mask,
                                 obj_points=obj_points,
                                 obj_occupancy=obj_occupancy,
                                 viz_cb=viz_cb)

        not_progressing_threshold = self.hparams['not_progressing_threshold']
        obj_transforms, viz_vars = iterative_projection(initial_value=initial_transformation_params,
                                                        target=target_transformation_params,
                                                        n=self.hparams['n_outer_iters'],
                                                        m=self.hparams['max_steps'],
                                                        step_towards_target=project_opt.step_towards_target,
                                                        project_opt=project_opt,
                                                        x_distance=self.scenario.aug_distance,
                                                        not_progressing_threshold=not_progressing_threshold,
                                                        device=device)

        transformation_matrices = self.scenario.transformation_params_to_matrices(obj_transforms, device)
        # NOTE: to_local_frame is [b, 3] but technically it should be [b, k, 3]
        obj_points_aug, to_local_frame = transform_obj_points(obj_points, moved_mask, transformation_matrices)

        is_valid = self.check_is_valid(moved_mask=moved_mask,
                                       obj_points_aug=obj_points_aug,
                                       obj_occupancy=obj_occupancy,
                                       extent=extent,
                                       sdf=project_opt.obj_sdf,
                                       sdf_aug=viz_vars.sdf_aug)
        return transformation_matrices, to_local_frame, is_valid

    def check_is_valid(self, moved_mask, obj_points_aug, obj_occupancy, extent, sdf, sdf_aug):
        """

        Args:
            moved_mask: [b, m]
            obj_points_aug: [b, m, T, n_points, 3]
            obj_occupancy:  [b, m, T, n_points]
            extent :[b, 4]
            sdf: the SDF values for the unaugmented points [b, m, T, n_points]
            sdf_aug:  the SDF values for the augmented points [b, m, T, n_points]

        Returns:

        """
        batch_size = moved_mask.shape[0]
        bbox_loss_batch = self.bbox_loss(obj_points_aug, extent).reshape([batch_size, -1]).max(dim=1)[0]
        bbox_constraint_satisfied = (bbox_loss_batch < self.hparams['max_bbox_violation']).float()

        if self.no_occupancy:
            env_constraints_satisfied = 1.0
        else:
            env_constraints_satisfied_ = check_env_constraints(obj_occupancy, sdf_aug)  # [b, m, T, n]
            num_env_constraints_violated = (1 - env_constraints_satisfied_).sum(dim=-1)
            num_env_constraints_violated = num_env_constraints_violated.sum(dim=-1)
            num_env_constraints_violated_moved = sum_over_moved(moved_mask, num_env_constraints_violated)
            env_constraints_satisfied = num_env_constraints_violated_moved < self.hparams['max_env_violations']
            env_constraints_satisfied = env_constraints_satisfied.float()

        if self.no_delta_min_dist:
            delta_min_dist_satisfied = 1.0
        else:
            delta_dist = torch.abs(sdf - sdf_aug)
            delta_min_dist = delta_dist.reshape([batch_size, -1]).min(dim=1)[0]
            delta_min_dist_satisfied = (delta_min_dist < self.hparams['delta_min_dist_threshold']).float()

        constraints_satisfied = env_constraints_satisfied * bbox_constraint_satisfied * delta_min_dist_satisfied
        return constraints_satisfied

    def sample_target_transform_params(self, batch_size: int, k_transforms: int, device):
        n_total = batch_size * k_transforms
        good_enough_percentile = self.hparams['good_enough_percentile']
        n_samples = int(1 / good_enough_percentile) * n_total

        target_params = self.scenario.sample_target_aug_params(self.rng, self.hparams, n_samples, device)

        # pick the most valid transforms, via the learned object state augmentation validity model
        best_target_params = pick_best_params(self.invariance_model_wrapper, target_params, batch_size)
        best_target_params = torch.reshape(best_target_params, [batch_size, k_transforms, target_params.shape[-1]])
        return best_target_params

    def use_original_if_invalid(self,
                                is_valid,
                                batch_size,
                                inputs,
                                inputs_aug,
                                keys_aug):
        for k in keys_aug:
            v = inputs_aug[k]
            iv = torch.reshape(is_valid, [batch_size] + [1] * (v.ndim - 1))
            inputs_aug[k] = iv * inputs_aug[k] + (1 - iv) * inputs[k]
        return inputs_aug

    def aug_apply_no_ik(self,
                        moved_mask,
                        transformation_matrices,
                        to_local_frame,
                        inputs,
                        batch_size,
                        device):
        return self.scenario.aug_apply_no_ik(moved_mask=moved_mask,
                                             m=transformation_matrices,
                                             to_local_frame=to_local_frame,
                                             inputs=inputs,
                                             batch_size=batch_size,
                                             device=device)

    def do_augmentation(self):
        return self.hparams is not None

    def bbox_loss(self, obj_points_aug, extent):
        """

        Args:
            obj_points_aug:  [b,m,n_points,3]
            extent:  [b,6]

        Returns:

        """
        extent = extent.reshape([-1, 3, 2])  # [b,3,2]
        extent_expanded = extent[:, None, None, None]
        lower_extent = extent_expanded[..., 0]  # [b,1,1,3]
        upper_extent = extent_expanded[..., 1]
        lower_extent_loss = torch.clamp(obj_points_aug - upper_extent, min=0)  # [b,m,n_points,3]
        upper_extent_loss = torch.clamp(lower_extent - obj_points_aug, min=0)
        bbox_loss = (lower_extent_loss + upper_extent_loss).sum(-1)  # [b,m,n_points]
        return self.hparams['bbox_weight'] * bbox_loss
