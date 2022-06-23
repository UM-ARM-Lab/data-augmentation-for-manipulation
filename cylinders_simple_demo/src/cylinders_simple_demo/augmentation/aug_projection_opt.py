from dataclasses import dataclass
from typing import Optional, Callable

import torch
import torch.optim.lr_scheduler

from cylinders_simple_demo.augmentation.aug_opt_utils import transform_obj_points, dpoint_to_dparams, mean_over_moved
from cylinders_simple_demo.utils.torch_geometry import batch_point_to_idx, homogeneous
from cylinders_simple_demo.utils.utils import empty_callable


@dataclass
class VizVars:
    obj_points_aug: torch.Tensor  # [b, m_objects, n_points, 3]
    to_local_frame: torch.Tensor  # [b, 3]
    min_dist_points_aug: torch.Tensor
    delta_min_dist_grad_dpoint: torch.Tensor
    attract_repel_dpoint: torch.Tensor
    sdf_aug: torch.Tensor


def batch_index_3d_idx(sdf, indices):
    """
    Args:
        sdf: [b, h, w, c]
        indices: [b1, ..., b2, 3]
    Returns: [b1, ..., b2]
    """
    assert sdf.shape[0] == indices.shape[0], "sdf and indices must have matching batch dimensions"
    sdf_i_indices, sdf_j_indices, sdf_k_indices = torch.unbind(indices, -1)
    batch_indices = torch.arange(sdf.shape[0])
    for _ in range(indices.ndim - 2):
        batch_indices = batch_indices.unsqueeze(-1)
    return sdf[batch_indices, sdf_i_indices, sdf_j_indices, sdf_k_indices]


class AugProjOpt:
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
        self.batch_indices = torch.arange(batch_size)
        self.batch_indices2 = torch.arange(batch_size)[:, None]
        self.batch_indices3 = torch.arange(batch_size)[:, None, None]
        self.batch_indices4 = torch.arange(batch_size)[:, None, None, None]
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
        self.m_objs = self.obj_points.shape[1]
        self.obj_sdf = self.batch_lookup_points(self.sdf, self.obj_points)  # [b, m_objects, T, n_points]
        obj_sdf_flat = self.obj_sdf.reshape([batch_size, self.m_objs, -1])  # [b, m_objects, T*n_points]
        # compute the closest points of all objects
        min_dist_all_objs, min_dist_all_objs_idx = obj_sdf_flat.min(2)
        # mask so the non-moved objects have really high distance
        min_dist_masked = min_dist_all_objs + (1 - self.moved_mask) * 1e6
        # now min over objects. min_dist_obj_idx are the indexes into the object dim (1)
        self.min_dist, self.min_dist_obj_idx = min_dist_masked.min(1)  # [b]
        # min_dist_tp_idx is the index into the flattened time+point dimension
        self.min_dist_tp_idx = min_dist_all_objs_idx[self.batch_indices, self.min_dist_obj_idx]  # [b]

        # viz hyperparameters
        viz_params = self.hparams.get('viz', {})
        self.viz_scale = viz_params.get('scale', 1.0)
        self.viz_arrow_scale = viz_params.get('arrow_scale', 1.0)
        self.viz_delta_min_dist_grad_scale = viz_params.get('delta_min_dist_grad_scale', 4.0)
        self.viz_grad_epsilon = viz_params.get('viz_grad_epsilon', 1e-6)

    def batch_lookup_points(self, sdf, points):
        res_expanded = self.res  # [b]
        for _ in range(points.ndim - 2):
            res_expanded = res_expanded.unsqueeze(-1)
        origin_point_expanded = self.origin_point  # [b, 3]
        for _ in range(points.ndim - 2):
            origin_point_expanded = origin_point_expanded.unsqueeze(-2)
        point_indices = batch_point_to_idx(points, res_expanded, origin_point_expanded)
        # clamp to be in bounds
        h = sdf.shape[1]
        w = sdf.shape[2]
        c = sdf.shape[3]
        bounds_ones = torch.ones_like(point_indices)
        lower = bounds_ones * 0
        upper = bounds_ones * torch.LongTensor([h, w, c]) - 1
        point_indices_valid = point_indices.clamp(lower, upper)
        return batch_index_3d_idx(sdf, point_indices_valid)

    def make_opt(self, parameters):
        opt = torch.optim.SGD([parameters], lr=self.hparams['step_size'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.hparams['lr_decay'])
        return opt, scheduler

    def forward(self, obj_transforms):
        s = self.aug_opt.scenario

        # obj_points is the set of points that define the object state, ie. the swept rope points.
        # the obj_points are in world frame, so obj_params is a transform in world frame
        # to compute the object state constraints loss we need to transform this during each forward pass
        # we also need to call apply_object_augmentation* at the end
        # to update the rest of the "state" which is input to the network
        transformation_matrices = s.transformation_params_to_matrices(obj_transforms)
        obj_points_aug, to_local_frame = transform_obj_points(self.obj_points,
                                                              self.moved_mask,
                                                              transformation_matrices)

        # compute repel and attract gradient via the SDF # FIXME: what about OOB?
        obj_sdf_aug = self.batch_lookup_points(self.sdf, obj_points_aug)  # [b, m_objs, T, n_points]
        obj_sdf_grad_aug = self.batch_lookup_points(self.sdf_grad, obj_points_aug)
        obj_occupancy_aug = (obj_sdf_aug < 0).float()
        obj_occupancy_aug_change = self.obj_occupancy - obj_occupancy_aug
        attract_repel_dpoint = obj_sdf_grad_aug * obj_occupancy_aug_change[..., None]
        attract_repel_dpoint = attract_repel_dpoint * self.hparams['sdf_grad_weight']  # [b,m_objs, T, n_points, 3]

        # and also the grad for preserving the min dist
        obj_points_aug_flat = obj_points_aug.reshape([self.batch_size, self.m_objs, -1, 3])
        min_dist_points_aug = obj_points_aug_flat[self.batch_indices, self.min_dist_obj_idx, self.min_dist_tp_idx]
        # min_dist_aug are the SDF values for the aug points corresponding to the origin min-dist points
        min_dist_aug = self.batch_lookup_points(self.sdf, min_dist_points_aug)

        delta_min_dist = self.min_dist - min_dist_aug
        delta_min_dist_grad_dpoint = self.batch_lookup_points(self.sdf_grad, min_dist_points_aug)
        delta_min_dist_grad_dpoint = delta_min_dist_grad_dpoint * delta_min_dist[..., None]  # [b,3]
        delta_min_dist_grad_dpoint = delta_min_dist_grad_dpoint * self.hparams['delta_min_dist_weight']  # [b,3]

        return VizVars(obj_points_aug=obj_points_aug,
                       to_local_frame=to_local_frame,
                       min_dist_points_aug=min_dist_points_aug,
                       delta_min_dist_grad_dpoint=delta_min_dist_grad_dpoint,
                       attract_repel_dpoint=attract_repel_dpoint,
                       sdf_aug=obj_sdf_aug)

    def project(self, _: int, opt: torch.optim.Optimizer, scheduler, obj_transforms: torch.nn.Parameter):
        s = self.aug_opt.scenario

        opt.zero_grad()

        v = self.forward(obj_transforms)

        invariance_loss = self.aug_opt.invariance_model_wrapper.evaluate(obj_transforms)  # [b, k_transforms]
        # when the constant is larger, this kills the gradient
        invariance_loss = torch.clamp(invariance_loss, min=self.hparams['invariance_threshold'])
        invariance_loss = self.hparams['invariance_weight'] * invariance_loss
        invariance_loss = invariance_loss.mean(-1)  # [b]

        bbox_loss_batch = self.aug_opt.bbox_loss(v.obj_points_aug, self.extent)  # [b,k,T,n]
        bbox_loss = bbox_loss_batch.sum(-1)
        bbox_loss = bbox_loss.sum(-1)
        bbox_loss = bbox_loss.mean(-1)  # [b]

        losses = [bbox_loss]
        if not self.aug_opt.no_invariance:
            losses.append(invariance_loss)
        losses_sum = sum(losses)
        loss = losses_sum.mean()

        # Compute the jacobian of the transformation. Here the transformation parameters have dimension p
        jacobian = s.aug_transformation_jacobian(obj_transforms)[:, :, None, None]  # [b,k,1,1,p,4,4]
        to_local_frame_moved_mean_expanded = v.to_local_frame[:, None, None, None, :]
        obj_points_local_frame = self.obj_points - to_local_frame_moved_mean_expanded  # [b,m_objects,T,n_points,3]
        obj_points_local_frame_h = homogeneous(obj_points_local_frame)[..., None, :, None]  # [b,m,T,n_points,1,4,1]
        dpoint_dparams_h = torch.squeeze(torch.matmul(jacobian, obj_points_local_frame_h),
                                         dim=-1)  # [b,m,T,n_points,p,4]
        dpoint_dparams = dpoint_dparams_h[..., :3]  # [b,m,T,n_points,p,3]
        dpoint_dparams = torch.permute(dpoint_dparams, [0, 1, 2, 3, 5, 4])  # [b, m, T, n_points, 3, p]

        # chain rule
        attract_repel_sdf_grad = dpoint_to_dparams(v.attract_repel_dpoint, dpoint_dparams)
        attract_repel_sdf_grad = attract_repel_sdf_grad.mean(-2).mean(-2)
        moved_attract_repel_sdf_grad_mean = mean_over_moved(self.moved_mask, attract_repel_sdf_grad)  # [b, p]
        # NOTE: this code isn't general enough to handle multiple transformations (k>1)
        moved_attract_repel_sdf_grad_mean = torch.unsqueeze(moved_attract_repel_sdf_grad_mean, dim=-2)

        dpoint_dparams_flat = dpoint_dparams.reshape([self.batch_size, self.m_objs, -1, 3, 3])
        min_dist_dpoint_dparams = dpoint_dparams_flat[self.batch_indices, self.min_dist_obj_idx, self.min_dist_tp_idx]
        delta_min_dist_grad_dparams = dpoint_to_dparams(v.delta_min_dist_grad_dpoint,  # [b, 3],
                                                        min_dist_dpoint_dparams)  # [b, 3, p] -> [b, p]
        delta_min_dist_grad_dparams = torch.unsqueeze(delta_min_dist_grad_dparams, dim=-2)

        loss.backward()
        obj_transforms_grad = obj_transforms.grad  # modify this gradient

        # combine with the gradient for the other aspects of the loss, those computed by torch.gradient
        if not self.aug_opt.no_occupancy:
            obj_transforms_grad += moved_attract_repel_sdf_grad_mean
        if not self.aug_opt.no_delta_min_dist:
            obj_transforms_grad += delta_min_dist_grad_dparams

        obj_transforms_grad = self.clip_env_aug_grad(obj_transforms_grad)

        obj_transforms.grad = obj_transforms_grad  # now apply and step
        opt.step()
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        can_terminate = self.can_terminate(lr, obj_transforms_grad)

        x_out = obj_transforms.clone()
        return x_out, can_terminate, v

    def clip_env_aug_grad(self, gradient):
        # we want grad_clip to be as close to in meters as possible, so here we scale by step size
        c = self.hparams['grad_clip'] / self.hparams['step_size']
        return torch.clip(gradient, -c, c)

    def can_terminate(self, lr, gradients):
        grad_norm = gradients.norm(dim=-1)
        step_size_i = grad_norm * lr
        can_terminate = step_size_i < self.hparams['step_size_threshold']
        all_can_terminate = torch.all(can_terminate)
        return all_can_terminate

    def step_towards_target(self, target_transforms, obj_transforms):
        # NOTE: although interpolating euler angles can be problematic or unintuitive,
        #  we have ensured the differences are <pi/2. So it should be ok
        x_interp = obj_transforms + (target_transforms - obj_transforms) * self.step_toward_target_fraction
        viz_vars = self.forward(x_interp)
        return x_interp, viz_vars

    def viz_func(self, _: Optional, obj_transforms, __, target, v: Optional[VizVars]):
        pass
