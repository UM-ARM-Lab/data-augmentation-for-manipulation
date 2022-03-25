from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchmeta.modules import MetaModule, MetaLinear, MetaSequential
from torchmeta.utils import gradient_update_parameters

from link_bot_pycommon.get_scenario import get_scenario
from moonshine.numpify import numpify
from moonshine.torch_utils import sequence_of_dicts_to_dict_of_tensors, vector_to_dict
from moonshine.torchify import torchify
from state_space_dynamics.udnn_torch import mask_after_first_0, compute_batch_time_loss


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
    def __init__(self, train_dataset: Optional, **hparams):
        super().__init__()

        if train_dataset is not None:
            max_example_idx = max([e['train']['example_idx'] for e in train_dataset])
            self.hparams['max_example_idx'] = max_example_idx
        else:
            max_example_idx = hparams['max_example_idx']

        self.save_hyperparameters(ignore=['train_dataset'])

        self.udnn = UDNN(**self.hparams)

        initial_sample_weights = torch.ones(max_example_idx + 1) * self.hparams.data_weight_init
        self.register_parameter("sample_weights", Parameter(initial_sample_weights))

        self.state_keys = self.udnn.state_keys
        self.state_metadata_keys = self.udnn.state_metadata_keys

        self.automatic_optimization = False

    def forward(self, inputs):
        return self.udnn.forward(inputs)

    def validation_step(self, inputs, batch_idx):
        meta_train_batch = inputs['meta_train']
        meta_train_udnn_outputs = self.udnn(meta_train_batch)
        meta_train_udnn_loss = self.udnn.compute_loss(meta_train_batch, meta_train_udnn_outputs)
        self.log('val_loss', meta_train_udnn_loss)

    def training_step(self, inputs, batch_idx):
        self.sample_weights.grad = None  # zero grad, very important!

        data_weight_opt, model_weight_opt = self.optimizers()

        train_batch = inputs['train']

        udnn_outputs = self.udnn(train_batch)
        udnn_loss = self.udnn.compute_batch_loss(train_batch, udnn_outputs)
        batch_indices = train_batch['example_idx']
        weights = torch.take_along_dim(self.sample_weights, batch_indices, dim=0)
        udnn_loss_weighted = torch.sum(udnn_loss * weights) / udnn_loss.nelement()  # inner loss

        self.log('train_loss', udnn_loss_weighted)

        # compute the update for udnn and get the updated params
        params = gradient_update_parameters(self.udnn,
                                            udnn_loss_weighted,
                                            step_size=self.hparams.udnn_inner_learning_rate,
                                            first_order=False)

        meta_train_batch = inputs['meta_train']
        meta_train_udnn_outputs = self.udnn(meta_train_batch, params=params)
        meta_train_udnn_loss = self.udnn.compute_loss(meta_train_batch, meta_train_udnn_outputs)
        meta_train_udnn_loss.backward()
        data_weight_opt.step()  # updates data weights

        # val_example_indices = meta_train_batch['example_idx']
        # val_sample_weights_sum = 0
        # n = 0
        # for train_batch_i, train_batch_example_idx in enumerate(train_batch['example_idx']):
        #     if train_batch_example_idx in val_example_indices.detach().cpu().numpy().tolist():
        #         val_sample_weights_sum += self.sample_weights[train_batch_example_idx]
        #         n += 1
        # if n > 0:
        #     avg_val_sample_weight = val_sample_weights_sum / n
        #     self.log('avg_val_sample_weight', avg_val_sample_weight)

        # same as the inner optimization just with adam, mostly for testing. I shouldn't really have to do it this way
        self.udnn.zero_grad()
        udnn_outputs = self.udnn(train_batch)
        udnn_loss = self.udnn.compute_batch_loss(train_batch, udnn_outputs)
        batch_indices = train_batch['example_idx']
        weights = torch.take_along_dim(self.sample_weights, batch_indices, dim=0)
        udnn_loss_weighted = torch.sum(udnn_loss * weights) / udnn_loss.nelement()
        udnn_loss_weighted.backward()
        model_weight_opt.step()  # updates model weights

        # now re-evaluate the validation loss
        meta_train_batch = inputs['meta_train']
        meta_train_udnn_outputs = self.udnn(meta_train_batch, params=params)
        meta_train_udnn_loss = self.udnn.compute_loss(meta_train_batch, meta_train_udnn_outputs)
        self.log('val_loss', meta_train_udnn_loss)

    def configure_optimizers(self):
        data_weight_opt = torch.optim.SGD([self.sample_weights], lr=self.hparams.weight_learning_rate)
        model_weight_opt = torch.optim.Adam(self.udnn.parameters(), lr=self.hparams.actual_udnn_learning_rate)
        return data_weight_opt, model_weight_opt