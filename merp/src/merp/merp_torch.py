from typing import Dict

import pytorch_lightning as pl
import torch
from torch import nn

from link_bot_data.dataset_utils import add_predicted_hack


class MERP(pl.LightningModule):

    def __init__(self, hparams, scenario):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.scenario = scenario

        self.mlp = nn.Sequential(nn.Linear(2 * 25 * 3, 1))
        self.has_checked_training_mode = False

    def forward(self, inputs: Dict[str, torch.Tensor]):
        if not self.has_checked_training_mode:
            self.has_checked_training_mode = True
            print(f"Training Mode? {self.training}")

        local_env, local_origin_point = self.get_local_env(inputs)

        voxel_grids = self.vg_info.make_voxelgrid_inputs(inputs, local_env, local_origin_point)

        conv_output = self.conv_encoder(voxel_grids)
        out_h = self.fc(inputs, conv_output)

        # for every timestep's output, map down to a single scalar, the logit for accept probability
        all_accept_logits = self.output_layer(out_h)
        # ignore the first output, it is meaningless to predict the validity of a single state
        valid_accept_logits = all_accept_logits[:, 1:]
        valid_accept_probabilities = self.sigmoid(valid_accept_logits)

        outputs = {
            'logits':        valid_accept_logits,
            'probabilities': valid_accept_probabilities,
            'out_h':         out_h,
        }

        return outputs

    def conv_encoder(self, voxel_grids, batch_size, time):
        conv_outputs_array = []
        for t in range(time):
            conv_z = voxel_grids[:, t]
            for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
                conv_h = conv_layer(conv_z)
                conv_z = pool_layer(conv_h)
            out_conv_z = conv_z
            out_conv_z_dim = out_conv_z.shape[1] * out_conv_z.shape[2] * out_conv_z.shape[3] * out_conv_z.shape[4]
            out_conv_z = torch.reshape(out_conv_z, [batch_size, out_conv_z_dim])
            conv_outputs_array.append(out_conv_z)
        conv_outputs = torch.stack(conv_outputs_array)
        conv_outputs = torch.permute(conv_outputs, [1, 0, 2])
        return conv_outputs

    def fc(self, input_dict, conv_output, training):
        states = {k: input_dict[add_predicted_hack(k)] for k in self.state_keys}
        states_in_local_frame = self.scenario.put_state_local_frame(states)
        actions = {k: input_dict[k] for k in self.action_keys}
        all_but_last_states = {k: v[:, :-1] for k, v in states.items()}
        actions = self.scenario.put_action_local_frame(all_but_last_states, actions)
        padded_actions = [torch.pad(v, [[0, 0], [0, 1], [0, 0]]) for v in actions.values()]
        if 'with_robot_frame' not in self.hparams:
            print("no hparam 'with_robot_frame'. This must be an old model!")
            concat_args = [conv_output] + list(states_in_local_frame.values()) + padded_actions
        elif self.hparams['with_robot_frame']:
            states_in_robot_frame = self.scenario.put_state_robot_frame(states)
            concat_args = ([conv_output] + list(states_in_robot_frame.values()) +
                           list(states_in_local_frame.values()) + padded_actions)
        else:
            concat_args = [conv_output] + list(states_in_local_frame.values()) + padded_actions

        concat_output = torch.cat(concat_args, axis=2)
        if self.hparams['batch_norm']:
            concat_output = self.batch_norm(concat_output, training=training)
        z = concat_output
        for dense_layer in self.dense_layers:
            z = dense_layer(z)
        out_d = z
        out_h = self.lstm(out_d)
        return out_h

    def get_local_env(self, inputs):
        state_0 = {k: inputs[add_predicted_hack(k)][:, 0] for k in self.state_keys}

        local_env_center = self.scenario.local_environment_center_differentiable(state_0)
        local_env, local_origin_point = self.local_env_helper.get(local_env_center, inputs, batch_size)

        return local_env, local_origin_point
    def compute_loss(self, inputs: Dict[str, torch.Tensor], outputs):
        return (outputs - inputs[add_predicted_hack('rope')].reshape(-1, 2 * 25 * 3).sum(-1,
                                                                                         keepdims=True)).square().sum()

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
