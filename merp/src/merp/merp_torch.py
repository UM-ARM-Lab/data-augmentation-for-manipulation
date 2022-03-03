from typing import Dict
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl
import torch

import rospy
from link_bot_data.dataset_utils import add_predicted_hack
from link_bot_data.local_env_helper import LocalEnvHelper
from link_bot_data.visualization import DebuggingViz
from link_bot_pycommon.grid_utils_np import environment_to_vg_msg
from moonshine import get_local_environment_torch
from moonshine.make_voxelgrid_inputs_torch import VoxelgridInfo
from moonshine.robot_points_torch import RobotVoxelgridInfo


class MERP(pl.LightningModule):

    def __init__(self, hparams, scenario):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.scenario = scenario

        self.local_env_h_rows = self.hparams['local_env_h_rows']
        self.local_env_w_cols = self.hparams['local_env_w_cols']
        self.local_env_c_channels = self.hparams['local_env_c_channels']
        self.point_state_keys_pred = [add_predicted_hack(k) for k in self.hparams['point_state_keys']]

        layers = []
        in_channels = 4
        for out_channels, kernel_size in self.hparams['conv_filters']:
            layers.append(nn.Conv3d(in_channels, out_channels, kernel_size))
            layers.append(nn.MaxPool3d(self.hparams['pooling']))

        # in_size = 10
        # for hidden_size in self.hparams['fc_layer_sizes']:
        #     layers.append(nn.Linear(in_size, hidden_size))
        #     layers.append(nn.ReLU())
        #     in_size = hidden_size

        final_hidden_dim = self.hparams['fc_layer_sizes'][-1]
        layers.append(nn.LSTM(final_hidden_dim, self.hparams['rnn_size'], 1))

        self.sequential = torch.nn.Sequential(*layers)

        self.output_layer = nn.Linear(final_hidden_dim, 1)

        self.debug = DebuggingViz(self.scenario, self.hparams.state_keys, self.hparams.action_keys)
        self.local_env_helper = LocalEnvHelper(h=self.local_env_h_rows, w=self.local_env_w_cols,
                                               c=self.local_env_c_channels,
                                               get_local_env_module=get_local_environment_torch)
        # TODO: use a dynamics model that is the UDNN torch + robot kinematics
        # self.robot_info = RobotVoxelgridInfo(joint_positions_key=add_predicted_hack('joint_positions'))
        self.vg_info = VoxelgridInfo(h=self.local_env_h_rows,
                                     w=self.local_env_w_cols,
                                     c=self.local_env_c_channels,
                                     state_keys=self.point_state_keys_pred,
                                     jacobian_follower=self.scenario.robot.jacobian_follower,
                                     robot_info=None,
                                     include_robot_geometry=False,
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

        b = 0
        for t in range(voxel_grids.shape[1]):
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
                self.debug.plot_state_rviz(inputs, b, t, 'inputs')

        # conv_output = self.conv_encoder(voxel_grids)
        # out_h = self.fc(inputs, conv_output)
        # all_accept_logits = F.sigmoid(out_h)
        #
        # # for every timestep's output, map down to a single scalar, the logit for accept probability
        # all_accept_logits = self.output_layer(out_h)
        # # ignore the first output, it is meaningless to predict the validity of a single state
        # valid_accept_logits = all_accept_logits[:, 1:]
        # valid_accept_probabilities = self.sigmoid(valid_accept_logits)
        #
        # outputs = {
        #     'logits':        valid_accept_logits,
        #     'probabilities': valid_accept_probabilities,
        #     'out_h':         out_h,
        # }
        #
        # return outputs

    # def conv_encoder(self, voxel_grids, batch_size, time):
    #     conv_outputs_array = []
    #     for t in range(time):
    #         conv_z = voxel_grids[:, t]
    #         for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
    #             conv_h = conv_layer(conv_z)
    #             conv_z = pool_layer(conv_h)
    #         out_conv_z = conv_z
    #         out_conv_z_dim = out_conv_z.shape[1] * out_conv_z.shape[2] * out_conv_z.shape[3] * out_conv_z.shape[4]
    #         out_conv_z = torch.reshape(out_conv_z, [batch_size, out_conv_z_dim])
    #         conv_outputs_array.append(out_conv_z)
    #     conv_outputs = torch.stack(conv_outputs_array)
    #     conv_outputs = torch.permute(conv_outputs, [1, 0, 2])
    #     return conv_outputs
    #
    # def fc(self, input_dict, conv_output, training):
    #     states = {k: input_dict[add_predicted_hack(k)] for k in self.hparams.state_keys}
    #     states_in_local_frame = self.scenario.put_state_local_frame(states)
    #     actions = {k: input_dict[k] for k in self.hparams.action_keys}
    #     all_but_last_states = {k: v[:, :-1] for k, v in states.items()}
    #     actions = self.scenario.put_action_local_frame(all_but_last_states, actions)
    #     padded_actions = [F.pad(v, [0, 0, 0, 1, 0, 0]) for v in actions.values()]
    #
    #     states_in_robot_frame = self.scenario.put_state_robot_frame(states)
    #     concat_args = ([conv_output] + list(states_in_robot_frame.values()) + list(
    #         states_in_local_frame.values()) + padded_actions)
    #
    #     concat_output = torch.cat(concat_args, 2)
    #     return out_h

    def get_local_env(self, inputs):
        batch_size = inputs['time_idx'].shape[0]
        state_0 = {k: inputs[add_predicted_hack(k)][:, 0] for k in self.hparams.point_state_keys}

        local_env_center = self.scenario.local_environment_center_differentiable_torch(state_0)
        local_env, local_origin_point = self.local_env_helper.get(local_env_center, inputs, batch_size)

        return local_env, local_origin_point

    def compute_loss(self, inputs: Dict[str, torch.Tensor], outputs):
        raise NotImplementedError()

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
