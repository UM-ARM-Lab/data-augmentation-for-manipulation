from typing import Dict

import pytorch_lightning as pl
import torch
from torch import nn
from torchmeta.modules import MetaModule, MetaLinear, MetaSequential
from torchmeta.utils import gradient_update_parameters

from link_bot_pycommon.get_scenario import get_scenario
from moonshine.numpify import numpify
from moonshine.torch_utils import sequence_of_dicts_to_dict_of_tensors, vector_to_dict
from moonshine.torchify import torchify
from state_space_dynamics.udnn_torch import mask_after_first_0, compute_batch_time_loss


class VNet(MetaModule):
    def __init__(self, **hparams):
        super(VNet, self).__init__()

        datset_params = hparams['dataset_hparams']
        data_collection_params = datset_params['data_collection_params']
        self.dataset_state_description: Dict = data_collection_params['state_description']
        self.state_description = {k: self.dataset_state_description[k] for k in hparams['state_keys']}
        self.total_state_dim = sum([self.dataset_state_description[k] for k in hparams['state_keys']])
        in_size = self.total_state_dim
        in_size = 1

        h = hparams['vnet']['h_dim']
        self.linear1 = MetaLinear(in_size, h)
        self.relu1 = nn.LeakyReLU()
        self.linear2 = MetaLinear(h, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        out = self.linear2(x)
        return torch.sigmoid(out)


class UDNN(MetaModule, pl.LightningModule):
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
            layers.append(MetaLinear(in_size, fc_layer_size))
            layers.append(torch.nn.ReLU())
            in_size = fc_layer_size
        layers.append(MetaLinear(fc_layer_size, self.total_state_dim))

        self.mlp = MetaSequential(*layers)

    def forward(self, inputs, params=None):
        if params is None:
            params = dict(self.named_parameters())

        actions = {k: inputs[k] for k in self.hparams.action_keys}
        input_sequence_length = actions[self.hparams.action_keys[0]].shape[1]
        s_0 = {k: inputs[k][:, 0] for k in self.hparams.state_keys}

        pred_states = [s_0]
        for t in range(input_sequence_length):
            s_t = pred_states[-1]
            action_t = {k: inputs[k][:, t] for k in self.hparams.action_keys}
            s_t_plus_1 = self.one_step_forward(action_t, s_t, params)

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

    def one_step_forward(self, action_t, s_t, params):
        local_action_t = self.scenario.put_action_local_frame(s_t, action_t)
        s_t_local = self.scenario.put_state_local_frame_torch(s_t)
        states_and_actions = list(s_t_local.values()) + list(local_action_t.values())
        z_t = torch.cat(states_and_actions, -1)

        # DEBUGGING
        # self.plot_local_state_action_rviz(local_action_t, s_t_local)

        z_t = self.mlp(z_t, params=self.get_subdict(params, 'mlp'))
        delta_s_t = vector_to_dict(self.state_description, z_t, self.device)
        s_t_plus_1 = self.scenario.integrate_dynamics(s_t, delta_s_t)
        return s_t_plus_1

    def compute_batch_loss(self, inputs, outputs, no_weights=True):
        batch_time_loss = compute_batch_time_loss(inputs, outputs)
        if no_weights:
            batch_loss = batch_time_loss.sum(-1)
        else:
            weights = self.get_weights(batch_time_loss, inputs)
            batch_loss = (batch_time_loss * weights).sum(-1)
        return batch_loss

    def compute_loss(self, inputs, outputs, no_weights=True):
        batch_loss = self.compute_batch_loss(inputs, outputs, no_weights=no_weights)
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


class MWNet(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        self.udnn = UDNN(**self.hparams)
        self.vnet = VNet(**self.hparams)

        self.state_keys = self.udnn.state_keys
        self.state_metadata_keys = self.udnn.state_metadata_keys

        self.automatic_optimization = False

    def forward(self, inputs):
        return self.udnn.forward(inputs)

    def training_step(self, inputs, batch_idx):
        train_batch = inputs['train']

        optimizer_model, optimizer_vnet = self.optimizers()

        udnn_outputs = self.udnn(train_batch)
        udnn_loss = self.udnn.compute_batch_loss(train_batch, udnn_outputs).unsqueeze(-1)
        weights = self.vnet(udnn_loss)

        udnn_loss_weighted = torch.sum(udnn_loss * weights) / udnn_loss.nelement()  # inner loss

        ex0_indices = torch.nonzero(1 - train_batch['example_idx']).squeeze()
        ex1_indices = torch.nonzero(train_batch['example_idx']).squeeze()
        self.log("ex0_pred_weight_mean", weights[ex0_indices].mean())
        self.log("ex1_pred_weight_mean", weights[ex1_indices].mean())

        self.log('udnn_loss_weighted', udnn_loss_weighted)

        # compute the update for udnn and get the updated params
        self.udnn.zero_grad()
        params = gradient_update_parameters(self.udnn,
                                            udnn_loss_weighted,
                                            step_size=self.hparams.vnet['udnn_inner_learning_rate'],
                                            first_order=False)

        meta_train_batch = inputs['meta_train']
        meta_train_udnn_outputs = self.udnn(meta_train_batch, params=params)
        meta_train_udnn_loss = self.udnn.compute_loss(meta_train_batch, meta_train_udnn_outputs)
        meta_train_udnn_loss.backward()  # outer loss
        optimizer_vnet.step()  # updates vnet
        self.log('udnn_meta_loss', meta_train_udnn_loss)

        self.udnn.load_state_dict(params)  # actually set the new weights for udnn

    def configure_optimizers(self):
        optimizer_model = torch.optim.SGD(self.udnn.parameters(),
                                          lr=self.hparams.learning_rate,
                                          momentum=0.9,
                                          weight_decay=5e-4)
        optimizer_vnet = torch.optim.Adam(self.vnet.parameters(),
                                          lr=self.hparams.vnet['learning_rate'],
                                          weight_decay=1e-4)
        return optimizer_model, optimizer_vnet
