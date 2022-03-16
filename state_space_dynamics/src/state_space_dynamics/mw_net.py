from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmeta.modules import MetaModule, MetaLinear

from link_bot_pycommon.get_scenario import get_scenario
from moonshine.numpify import numpify
from moonshine.torch_utils import sequence_of_dicts_to_dict_of_tensors, vector_to_dict
from moonshine.torchify import torchify
from state_space_dynamics.udnn_torch import UDNN


# class VNet(pl.LightningModule):
#
#     def __init__(self, **hparams):
#         super().__init__()
#         self.save_hyperparameters()
#
#         datset_params = self.hparams['dataset_hparams']
#         data_collection_params = datset_params['data_collection_params']
#         self.dataset_state_description: dict = data_collection_params['state_description']
#         self.state_description = {k: self.dataset_state_description[k] for k in self.hparams.state_keys}
#         self.total_state_dim = sum([self.dataset_state_description[k] for k in self.hparams.state_keys])
#         in_size = self.total_state_dim
#
#         h = self.hparams['h']
#         self.mlp = nn.Sequential(
#             nn.Linear(in_size, h),
#             nn.LeakyReLU(),
#             nn.Linear(h, h),
#             nn.LeakyReLU(),
#             nn.Linear(h, 1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         return self.mlp(x)


class VNet(MetaModule):
    def __init__(self, **hparams):
        super(VNet, self).__init__()

        datset_params = hparams['dataset_hparams']
        data_collection_params = datset_params['data_collection_params']
        self.dataset_state_description: Dict = data_collection_params['state_description']
        self.state_description = {k: self.dataset_state_description[k] for k in hparams['state_keys']}
        self.total_state_dim = sum([self.dataset_state_description[k] for k in hparams['state_keys']])
        in_size = self.total_state_dim

        h = hparams['vnet']['h_dim']
        self.linear1 = MetaLinear(in_size, h)
        self.relu1 = nn.ReLU()
        self.linear2 = MetaLinear(h, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return F.sigmoid(out)


class UDNN(MetaModule):
    def __init__(self, with_joint_positions=False, **hparams):
        super().__init__()
        self.save_hyperparameters()

        datset_params = self.hparams['dataset_hparams']
        data_collection_params = datset_params['data_collection_params']
        self.scenario = get_scenario(self.hparams.scenario, params=data_collection_params['scenario_params'])
        # FIXME: this dict is currently not getting generated for the newly collected datasets :(
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

    def compute_batch_time_loss(self, inputs, outputs):
        loss_by_key = []
        for k, y_pred in outputs.items():
            y_true = inputs[k]
            # mean over time and state dim but not batch, not yet.
            loss = (y_true - y_pred).square().mean(dim=-1)
            loss_by_key.append(loss)
        batch_time_loss = torch.stack(loss_by_key).mean(dim=0)
        return batch_time_loss

    def compute_batch_loss(self, inputs, outputs, no_weights=True):
        """

        Args:
            inputs:
            outputs:
            no_weights: Ignore the weight in the "inputs"

        Returns:

        """
        batch_time_loss = self.compute_batch_time_loss(inputs, outputs)
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


class MWNet(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        self.udnn = UDNN(**self.hparams)
        self.vnet = VNet(**self.hparams)
        self.meta_udnn = UDNN(**self.hparams)

        self.state_keys = self.udnn.state_keys
        self.state_metadata_keys = self.udnn.state_metadata_keys

        self.automatic_optimization = False

    def forward(self, inputs):
        return self.udnn.forward(inputs)

    def training_step(self, inputs, batch_idx):
        train_batch = inputs['train']

        optimizer_model, optimizer_vnet = self.optimizers()

        # First we copy the dynamics model and run a gradient descent step using the weights net
        # then we backprop through that entire gradient descent step to update the weights net
        # the weights net is optimized on un-weighted performance on the meta-training set
        # finally, we update the dynamics model using the weighted loss onf the training set
        self.meta_udnn.load_state_dict(self.udnn.state_dict())

        # the naming convention is:
        # A_B
        # meta_A_B
        # meta_A_meta_B
        # A_meta_B
        # the first meta applies to A the second meta applies to B
        meta_udnn_outputs = self.meta_udnn(train_batch)
        meta_udnn_loss = self.meta_udnn.compute_batch_time_point_loss(train_batch,
                                                                      meta_udnn_outputs)  # [b, t, total_state_dim]

        weights = self.vnet(meta_udnn_loss)
        batch_time_size = (meta_udnn_loss.shape[0] + meta_udnn_loss.shape[1])
        meta_udnn_loss_weighted = torch.sum(meta_udnn_loss * weights) / batch_time_size

        # perform a gradient descent step on meta_udnn in a way that lets us backprop through, to the params of vnet
        self.meta_udnn.zero_grad()
        grads = torch.autograd.grad(meta_udnn_loss_weighted, self.meta_udnn.parameters(), create_graph=True)
        for (param_name, param_value), grad in zip(self.meta_udnn.named_parameters(), grads):
            setattr(self.udnn, param_name, param_value - self.hparams.learning_rate * grad)

        # now use the meta data batch, pass that through the meta_udnn, compute loss, then backprop to update VNet
        meta_train_batch = inputs['meta_train']
        meta_udnn_meta_outputs = self.meta_udnn(meta_train_batch)
        meta_loss_meta_batch = self.meta_udnn.compute_loss(meta_train_batch, meta_udnn_meta_outputs)
        optimizer_vnet.zero_grad()
        meta_loss_meta_batch.backward()
        optimizer_vnet.step()
        print(self.vnet.linear1.weight[0, 0])

        # now update the udnn weights. Pass the batch through udnn, compute loss, backprop to update udnn
        udnn_outputs = self.udnn(train_batch)
        udnn_loss = self.udnn.compute_batch_time_point_loss(train_batch, udnn_outputs)
        with torch.no_grad():  # we don't want to update the weights of vnet here, that happened above
            meta_weights = self.vnet(udnn_loss)
        udnn_loss_weighted = torch.sum(udnn_loss * meta_weights) / batch_time_size

        # Question -- why not optimize this first than back-prop through this to do the outer loss for vnet?
        optimizer_model.zero_grad()
        udnn_loss_weighted.backward()
        optimizer_model.step()

        # finally, just run the udnn on the meta training set without any backprop, just for logging
        udnn_meta_outputs = self.udnn(meta_train_batch)
        udnn_meta_loss = self.udnn.compute_loss(meta_train_batch, udnn_meta_outputs, no_weights=True)

        self.log('udnn_meta_loss', udnn_meta_loss)
        self.log('udnn_loss_weighted', udnn_loss_weighted)

    def configure_optimizers(self):
        optimizer_model = torch.optim.SGD(self.udnn.parameters(),
                                          lr=self.hparams.learning_rate,
                                          momentum=0.9,
                                          weight_decay=5e-4)
        optimizer_vnet = torch.optim.Adam(self.vnet.parameters(),
                                          lr=1e-3,
                                          weight_decay=1e-4)
        return optimizer_model, optimizer_vnet
