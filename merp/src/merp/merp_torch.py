from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

import rospy
from link_bot_data.dataset_utils import add_predicted_hack
from link_bot_data.local_env_helper import LocalEnvHelper
from link_bot_data.visualization import DebuggingViz
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.grid_utils_np import environment_to_vg_msg
from moonshine import get_local_environment_torch
from moonshine.make_voxelgrid_inputs_torch import VoxelgridInfo
from moonshine.robot_points_torch import RobotVoxelgridInfo


def debug_vgs():
    return rospy.get_param("DEBUG_VG", False)


class MERP(pl.LightningModule):

    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        datset_params = self.hparams['dataset_hparams']
        data_collection_params = datset_params['data_collection_params']
        self.scenario = get_scenario(self.hparams.scenario, params=data_collection_params['scenario_params'])

        self.local_env_h_rows = self.hparams['local_env_h_rows']
        self.local_env_w_cols = self.hparams['local_env_w_cols']
        self.local_env_c_channels = self.hparams['local_env_c_channels']
        self.point_state_keys_pred = [add_predicted_hack(k) for k in self.hparams['point_state_keys']]

        conv_layers = []
        in_channels = 4
        for out_channels, kernel_size in self.hparams['conv_filters']:
            conv_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size))
            conv_layers.append(nn.MaxPool3d(self.hparams['pooling']))
            in_channels = out_channels

        fc_layers = []
        state_desc = data_collection_params['state_description']
        action_desc = data_collection_params['action_description']
        state_size = sum([state_desc[k] for k in self.hparams.state_keys])
        action_size = sum([action_desc[k] for k in self.hparams.action_keys])

        conv_out_size = int(self.hparams['conv_filters'][-1][0] * np.prod(self.hparams['conv_filters'][-1][1]))
        in_size = conv_out_size + 2 * state_size + action_size
        for hidden_size in self.hparams['fc_layer_sizes']:
            fc_layers.append(nn.Linear(in_size, hidden_size))
            fc_layers.append(nn.ReLU())
            in_size = hidden_size
        final_hidden_dim = self.hparams['fc_layer_sizes'][-1]
        fc_layers.append(nn.LSTM(final_hidden_dim, self.hparams['rnn_size'], 1))

        self.conv_encoder = torch.nn.Sequential(*conv_layers)
        self.fc = torch.nn.Sequential(*fc_layers)

        self.output_layer = nn.Linear(final_hidden_dim, 1)

        self.debug = DebuggingViz(self.scenario, self.hparams.state_keys, self.hparams.action_keys)
        self.local_env_helper = LocalEnvHelper(h=self.local_env_h_rows, w=self.local_env_w_cols,
                                               c=self.local_env_c_channels,
                                               get_local_env_module=get_local_environment_torch)
        self.robot_info = RobotVoxelgridInfo(joint_positions_key=add_predicted_hack('joint_positions'))
        self.vg_info = VoxelgridInfo(h=self.local_env_h_rows,
                                     w=self.local_env_w_cols,
                                     c=self.local_env_c_channels,
                                     state_keys=self.point_state_keys_pred,
                                     jacobian_follower=self.scenario.robot.jacobian_follower,
                                     robot_info=self.robot_info,
                                     include_robot_geometry=True,
                                     scenario=self.scenario,
                                     )

        self.has_checked_training_mode = False

    def forward(self, inputs: Dict[str, torch.Tensor]):
        if not self.has_checked_training_mode:
            self.has_checked_training_mode = True
            print(f"Training Mode? {self.training}")

        if self.local_env_helper.device != self.device:
            self.local_env_helper.to(self.device)

        local_env, local_origin_point = self.get_local_env(inputs)

        batch_size, time = inputs['time_idx'].shape[0:2]
        voxel_grids = self.vg_info.make_voxelgrid_inputs(inputs, local_env, local_origin_point, batch_size, time)

        if debug_vgs():
            b = 0
            for t in range(voxel_grids.shape[1]):
                self.debug.plot_pred_state_rviz(inputs, b, t, 'pred_inputs')
                for i in range(voxel_grids.shape[-1]):
                    raster_dict = {
                        'env':          voxel_grids[b, t, :, :, :, i].cpu().numpy(),
                        'res':          inputs['res'][b].cpu().numpy(),
                        'origin_point': local_origin_point[b].cpu().numpy(),
                    }

                    self.scenario.send_occupancy_tf(raster_dict, parent_frame_id='robot_root',
                                                    child_frame_id='local_env_vg')
                    raster_msg = environment_to_vg_msg(raster_dict, frame='local_env_vg', stamp=rospy.Time.now())
                    self.debug.raster_debug_pubs[i].publish(raster_msg)

        states = {k: inputs[add_predicted_hack(k)] for k in self.hparams.state_keys}
        states_local_frame = self.scenario.put_state_local_frame_torch(states)
        actions = {k: inputs[k] for k in self.hparams.action_keys}
        all_but_last_states = {k: v[:, :-1] for k, v in states.items()}
        actions = self.scenario.put_action_local_frame(all_but_last_states, actions)
        padded_actions = [F.pad(v, [0, 0, 0, 1, 0, 0]) for v in actions.values()]

        states_robot_frame = self.scenario.put_state_robot_frame(states)

        flat_voxel_grids = voxel_grids.reshape(
            [-1, 4, self.local_env_h_rows, self.local_env_w_cols, self.local_env_c_channels])
        flat_conv_h = self.conv_encoder(flat_voxel_grids)
        conv_h = flat_conv_h.reshape(batch_size, time, -1)

        cat_args = [conv_h] + list(states_robot_frame.values()) + list(states_local_frame.values()) + padded_actions
        fc_in = torch.cat(cat_args, -1)
        out_h, _ = self.fc(fc_in)

        # for every timestep's output, map down to a single scalar, the logit for accept probability
        predicted_errors = self.output_layer(out_h)

        # ignore the first output, it is meaningless to predict the error at the "start" state
        predicted_error = predicted_errors[:, 1:].squeeze(1).squeeze(1)

        return predicted_error

    def get_local_env(self, inputs):
        batch_size = inputs['time_idx'].shape[0]
        state_0 = {k: inputs[add_predicted_hack(k)][:, 0] for k in self.hparams.point_state_keys}

        local_env_center = self.scenario.local_environment_center_differentiable_torch(state_0)
        local_env, local_origin_point = self.local_env_helper.get(local_env_center, inputs, batch_size)

        return local_env, local_origin_point

    def compute_loss(self, inputs: Dict[str, torch.Tensor], outputs):
        error_after = inputs['error'][:, 1]
        loss = F.mse_loss(outputs, error_after)
        return loss

    def training_step(self, train_batch: Dict[str, torch.Tensor], batch_idx):
        outputs = self.forward(train_batch)
        loss = self.compute_loss(train_batch, outputs)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch: Dict[str, torch.Tensor], batch_idx):
        outputs = self.forward(val_batch)
        loss = self.compute_loss(val_batch, outputs)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
