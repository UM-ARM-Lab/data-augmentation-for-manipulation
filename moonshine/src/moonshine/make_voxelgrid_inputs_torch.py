from dataclasses import dataclass
from typing import List, Dict

import pyjacobian_follower
import torch

from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.geometry_torch import densify_points
from moonshine.numpify import numpify
from moonshine.raster_3d_torch import points_to_voxel_grid_res_origin_point_batched
from moonshine.robot_points_torch import batch_transform_robot_points, RobotVoxelgridInfo, \
    batch_robot_state_to_transforms


@dataclass
class VoxelgridInfo:
    h: int
    w: int
    c: int
    state_keys: List[str]
    jacobian_follower: pyjacobian_follower.JacobianFollower
    robot_info: RobotVoxelgridInfo
    include_robot_geometry: bool
    scenario: ScenarioWithVisualization

    def make_voxelgrid_inputs(self, inputs: Dict, local_env, local_origin_point, batch_size, time, viz: bool = False):
        local_voxel_grids = []
        for t in range(time):
            local_voxel_grid_t = self.make_voxelgrid_inputs_t(inputs, local_env, local_origin_point, t, batch_size, viz)
            local_voxel_grids.append(local_voxel_grid_t)
        local_voxel_grids = torch.stack(local_voxel_grids, 1)
        return local_voxel_grids

    def make_voxelgrid_inputs_t(self,
                                inputs,
                                local_env,
                                local_origin_point,
                                t,
                                batch_size,
                                viz_points: bool = False):
        state_t = {k: inputs[k][:, t] for k in self.state_keys}
        local_voxel_grid_t = [local_env]
        device = local_env.device

        # insert the rastered states
        for (k, state_component_t) in state_t.items():
            points = state_component_t.reshape([batch_size, -1, 3])
            num_densify = 5
            points = densify_points(batch_size=batch_size, points=points, num_densify=num_densify)
            # self.scenario.plot_points_rviz(points[2], label='dense')
            state_component_voxel_grid = points_to_voxel_grid_res_origin_point_batched(points,
                                                                                       inputs['res'],
                                                                                       local_origin_point,
                                                                                       self.h,
                                                                                       self.w,
                                                                                       self.c,
                                                                                       batch_size)
            local_voxel_grid_t.append(state_component_voxel_grid)

        if self.include_robot_geometry:
            robot_points = self.make_robot_points_batched(batch_size, inputs, t, viz_points).to(
                device)  # [b, n_points, 3]
            robot_voxel_grid = points_to_voxel_grid_res_origin_point_batched(robot_points,
                                                                             inputs['res'],
                                                                             local_origin_point,
                                                                             self.h,
                                                                             self.w,
                                                                             self.c,
                                                                             batch_size)
            local_voxel_grid_t.append(robot_voxel_grid)

        local_voxel_grid_t = torch.stack(local_voxel_grid_t, 1)
        return local_voxel_grid_t

    def make_robot_points_batched(self, batch_size, inputs, t, viz_points: bool = False):
        names = inputs['joint_names'][:, t]
        positions = inputs[self.robot_info.joint_positions_key][:, t]
        link_to_robot_transforms = batch_robot_state_to_transforms(self.jacobian_follower,
                                                                   numpify(names),
                                                                   positions.detach().cpu().numpy(),
                                                                   self.robot_info.link_names)
        robot_points = batch_transform_robot_points(link_to_robot_transforms, self.robot_info, batch_size)
        if viz_points:
            self.scenario.plot_points_rviz(robot_points[0], label='robot_points', frame_id='robot_root')
        return robot_points
