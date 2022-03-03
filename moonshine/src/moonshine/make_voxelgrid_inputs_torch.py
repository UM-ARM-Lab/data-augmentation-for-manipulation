from dataclasses import dataclass
from typing import List, Dict

import pyjacobian_follower
import torch

from moonshine.robot_points_torch import batch_transform_robot_points, RobotVoxelgridInfo, batch_robot_state_to_transforms
from moonshine.geometry_torch import densify_points
from moonshine.numpify import numpify
from moonshine.raster_3d_tf import points_to_voxel_grid_res_origin_point_batched


@dataclass
class VoxelgridInfo:
    h: int
    w: int
    c: int
    state_keys: List[str]
    jacobian_follower: pyjacobian_follower.JacobianFollower
    robot_info: RobotVoxelgridInfo
    include_robot_geometry: bool

    def make_voxelgrid_inputs(self, input_dict: Dict, local_env, local_origin_point, batch_size, time):
        local_voxel_grids = []
        for t in range(time):
            local_voxel_grid_t = self.make_voxelgrid_inputs_t(input_dict, local_env, local_origin_point, t,
                                                              batch_size)

            local_voxel_grids.append(local_voxel_grid_t)

        local_voxel_grids = torch.stack(local_voxel_grids)
        return local_voxel_grids

    def make_voxelgrid_inputs_t(self,
                                inputs,
                                local_env,
                                local_origin_point,
                                t,
                                batch_size):
        state_t = {k: inputs[k][:, t] for k in self.state_keys}
        local_voxel_grid_t = [local_env]
        # insert the rastered states
        for (k, state_component_t) in state_t.items():
            points = state_component_t.reshape([batch_size, -1, 3])
            num_densify = 5
            points = densify_points(batch_size=batch_size, points=points, num_densify=num_densify)
            n_points_in_component = max(int(state_component_t.shape[1] / 3 * num_densify) - num_densify, 1)
            flat_points = torch.reshape(points, [-1, 3])
            flat_batch_indices = torch.repeat(torch.arange(batch_size, dtype=torch.int64), n_points_in_component, axis=0)
            flat_res = torch.repeat(inputs['res'], n_points_in_component, axis=0)
            flat_origin_point = torch.repeat(local_origin_point, n_points_in_component, axis=0)
            state_component_voxel_grid = points_to_voxel_grid_res_origin_point_batched(flat_batch_indices,
                                                                                       flat_points,
                                                                                       flat_res,
                                                                                       flat_origin_point,
                                                                                       self.h,
                                                                                       self.w,
                                                                                       self.c,
                                                                                       batch_size)
            local_voxel_grid_t.append(state_component_voxel_grid)

        if self.include_robot_geometry:
            robot_voxel_grid = self.make_robot_voxelgrid(inputs, local_origin_point, t, batch_size)
            local_voxel_grid_t.append(robot_voxel_grid)
            n_channels = len(self.state_keys) + 2
        else:
            n_channels = len(self.state_keys) + 1

        local_voxel_grid_t = torch.permute(torch.stack(local_voxel_grid_t), [1, 2, 3, 4, 0])
        return local_voxel_grid_t

    def make_robot_voxelgrid(self, inputs, local_origin_point, t, batch_size):
        robot_points = self.make_robot_points_batched(batch_size, inputs, t)
        n_robot_points = robot_points.shape[1]
        flat_batch_indices = torch.repeat(torch.arange(batch_size, dtype=torch.int64), n_robot_points, axis=0)
        flat_points = torch.reshape(robot_points, [-1, 3])
        flat_res = torch.repeat(inputs['res'], n_robot_points, axis=0)
        flat_origin_point = torch.repeat(local_origin_point, n_robot_points, axis=0)
        robot_voxelgrid = points_to_voxel_grid_res_origin_point_batched(flat_batch_indices,
                                                                        flat_points,
                                                                        flat_res,
                                                                        flat_origin_point,
                                                                        self.h,
                                                                        self.w,
                                                                        self.c,
                                                                        batch_size)
        return robot_voxelgrid

    def make_robot_points_batched(self, batch_size, inputs, t):
        names = inputs['joint_names'][:, t]
        positions = inputs[self.robot_info.joint_positions_key][:, t]
        link_to_robot_transforms = batch_robot_state_to_transforms(self.jacobian_follower,
                                                                   numpify(names),
                                                                   positions.numpy(),
                                                                   self.robot_info.link_names)
        robot_points = batch_transform_robot_points(link_to_robot_transforms, self.robot_info, batch_size)
        return robot_points
