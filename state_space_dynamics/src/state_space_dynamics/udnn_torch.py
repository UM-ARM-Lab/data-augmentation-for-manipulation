from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch

from link_bot_pycommon.get_scenario import get_scenario
from moonshine.numpify import numpify
from moonshine.torch_utils import vector_to_dict, sequence_of_dicts_to_dict_of_tensors
from moonshine.torchify import torchify


def mask_after_first_0(x):
    # TODO: vectorize?
    x_out = torch.cat([x, torch.zeros([x.shape[0], 1]).to(x.device)], -1)
    for b in range(x.shape[0]):
        i = x_out[b].argmin()
        x_out[b, i:] = 0
    return x_out[:, :-1]


def compute_batch_time_loss(inputs, outputs):
    loss_by_key = []
    for k, y_pred in outputs.items():
        y_true = inputs[k]
        # mean over time and state dim but not batch, not yet.
        loss = (y_true - y_pred).square().mean(-1)
        loss_by_key.append(loss)
    batch_time_loss = torch.stack(loss_by_key).mean(0)
    return batch_time_loss


class UDNN(pl.LightningModule):
    def __init__(self, with_joint_positions=False, **hparams):
        super().__init__()
        self.save_hyperparameters()

        datset_params = self.hparams['dataset_hparams']
        data_collection_params = datset_params['data_collection_params']
        self.scenario = get_scenario(self.hparams.scenario, params=data_collection_params['scenario_params'])
        self.dataset_state_description: Dict = data_collection_params['state_description']
        self.dataset_action_description: Dict = data_collection_params['action_description']
        self.state_keys = self.hparams.state_keys
        self.state_metadata_keys = self.hparams.state_metadata_keys
        self.state_description = {k: self.dataset_state_description[k] for k in self.hparams.state_keys}
        self.total_state_dim = sum([self.dataset_state_description[k] for k in self.hparams.state_keys])
        self.total_action_dim = sum([self.dataset_action_description[k] for k in self.hparams.action_keys])
        self.with_joint_positions = with_joint_positions

        in_size = self.total_state_dim + self.total_action_dim
        fc_layer_size = None

        layers = []
        for fc_layer_size in self.hparams.fc_layer_sizes:
            layers.append(torch.nn.Linear(in_size, fc_layer_size))
            layers.append(torch.nn.ReLU())
            in_size = fc_layer_size
        layers.append(torch.nn.Linear(fc_layer_size, self.total_state_dim))

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, inputs):
        actions = {k: inputs[k] for k in self.hparams.action_keys}
        input_sequence_length = actions[self.hparams.action_keys[0]].shape[1]
        s_0 = {k: inputs[k][:, 0] for k in self.hparams.state_keys}

        pred_states = [s_0]
        for t in range(input_sequence_length):
            s_t = pred_states[-1]
            action_t = {k: inputs[k][:, t] for k in self.hparams.action_keys}
            s_t_plus_1 = self.one_step_forward(action_t, s_t)

            pred_states.append(s_t_plus_1)

        pred_states_dict = sequence_of_dicts_to_dict_of_tensors(pred_states, axis=1)

        if self.with_joint_positions:
            # no need to do this during training, only during prediction/evaluation/testing
            inputs_np = numpify(inputs)
            inputs_np['batch_size'] = inputs['time_idx'].shape[0]
            _, joint_positions, joint_names = self.scenario.follow_jacobian_from_example(inputs_np,
                                                                                         j=self.scenario.robot.jacobian_follower)
            pred_states_dict['joint_positions'] = torchify(joint_positions).float()
            pred_states_dict['joint_names'] = joint_names

        return pred_states_dict

    def one_step_forward(self, action_t, s_t):
        local_action_t = self.scenario.put_action_local_frame(s_t, action_t)
        s_t_local = self.scenario.put_state_local_frame_torch(s_t)
        states_and_actions = list(s_t_local.values()) + list(local_action_t.values())
        z_t = torch.cat(states_and_actions, -1)

        # DEBUGGING
        # self.plot_local_state_action_rviz(local_action_t, s_t_local)

        z_t = self.mlp(z_t)
        delta_s_t = vector_to_dict(self.state_description, z_t, self.device)
        s_t_plus_1 = self.scenario.integrate_dynamics(s_t, delta_s_t)
        return s_t_plus_1

    def plot_local_state_action_rviz(self, local_action_t, s_t_local):
        self.scenario.plot_arrow_rviz(np.array([0, 0, 0]),
                                      local_action_t['left_gripper_delta'][0].cpu().detach().numpy(),
                                      label='left_action')
        self.scenario.plot_arrow_rviz(np.array([0, 0, 0]),
                                      local_action_t['right_gripper_delta'][0].cpu().detach().numpy(),
                                      label='right_action')
        local_rope = np.concatenate((s_t_local['left_gripper'][0].cpu().detach().numpy(),
                                     s_t_local['right_gripper'][0].cpu().detach().numpy(),
                                     s_t_local['rope'][0].cpu().detach().numpy()))
        local_rope_points = local_rope.reshape([27, 3])
        self.scenario.plot_points_rviz(local_rope_points, label='local_rope_points')

    def compute_batch_loss(self, inputs, outputs, no_weights=True):
        """

        Args:
            inputs:
            outputs:
            no_weights: Ignore the weight in the "inputs"

        Returns:

        """
        batch_time_loss = compute_batch_time_loss(inputs, outputs)
        if no_weights:
            batch_loss = batch_time_loss.sum(-1)
        else:
            weights = self.get_weights(batch_time_loss, inputs)
            batch_loss = (batch_time_loss * weights).sum(-1)
        return batch_loss

    def compute_loss(self, inputs, outputs, no_weights=True):
        batch_loss = self.compute_batch_loss(inputs, outputs, no_weights=no_weights)
        print(inputs['example_idx'])
        print(batch_loss)
        return batch_loss.mean()

    def compute_batch_time_point_loss(self, inputs, outputs):
        loss_by_key = []
        for k, y_pred in outputs.items():
            y_true = inputs[k]
            loss = (y_true - y_pred).square()
            loss_by_key.append(loss)
        batch_time_point_loss = torch.cat(loss_by_key, -1)
        return batch_time_point_loss

    def get_weights(self, batch_time_loss, inputs):
        if 'weight' in inputs:
            weights = inputs['weight']
        else:
            weights = torch.ones_like(batch_time_loss).to(self.device)
        weights = mask_after_first_0(weights)
        return weights

    def training_step(self, train_batch, batch_idx):
        outputs = self.forward(train_batch)
        loss = self.compute_loss(train_batch, outputs)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        outputs = self.forward(val_batch)
        loss = self.compute_loss(val_batch, outputs)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
