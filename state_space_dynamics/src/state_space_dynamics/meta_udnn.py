from copy import deepcopy
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from torchmeta.modules import MetaModule, MetaLinear, MetaSequential

from link_bot_pycommon.get_scenario import get_scenario
from moonshine.numpify import numpify
from moonshine.torch_utils import sequence_of_dicts_to_dict_of_tensors, vector_to_dict
from moonshine.torchify import torchify


def segment_lengths(inputs):
    initial_rope_points = inputs['rope'].reshape(inputs['rope'].shape[:-1] + (25, 3))
    initial_rope_segment_lengths = (initial_rope_points[..., 1:, :] - initial_rope_points[..., :-1, :]).norm(dim=-1)
    return initial_rope_segment_lengths


class UDNN(MetaModule, pl.LightningModule):
    def __init__(self, with_joint_positions=False, **hparams):
        super().__init__()
        self.save_hyperparameters()

        datset_params = self.hparams['dataset_hparams']
        self.data_collection_params = datset_params['data_collection_params']
        self.scenario = get_scenario(self.hparams.scenario, params=self.data_collection_params['scenario_params'])
        self.dataset_state_description: Dict = self.data_collection_params['state_description']
        self.dataset_action_description: Dict = self.data_collection_params['action_description']
        self.state_keys = self.hparams.state_keys
        self.state_metadata_keys = self.hparams.state_metadata_keys
        self.state_description = {k: self.dataset_state_description[k] for k in self.hparams.state_keys}
        self.total_state_dim = sum([self.dataset_state_description[k] for k in self.hparams.state_keys])
        self.total_action_dim = sum([self.dataset_action_description[k] for k in self.hparams.action_keys])
        self.with_joint_positions = with_joint_positions
        self.max_step_size = self.data_collection_params.get('max_step_size', 0.01)  # default for current rope sim

        in_size = self.total_state_dim + self.total_action_dim
        fc_layer_size = None

        layers = []
        for fc_layer_size in self.hparams.fc_layer_sizes:
            layers.append(MetaLinear(in_size, fc_layer_size))
            layers.append(torch.nn.ReLU())
            in_size = fc_layer_size
        layers.append(MetaLinear(fc_layer_size, self.total_state_dim))

        self.mlp = MetaSequential(*layers)

        self.val_model_errors = None
        self.test_model_errors = None

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

        z_t = self.mlp(z_t, params=self.get_subdict(params, 'mlp'))
        delta_s_t = vector_to_dict(self.state_description, z_t, self.device)
        s_t_plus_1 = self.scenario.integrate_dynamics(s_t, delta_s_t)
        return s_t_plus_1

    def compute_batch_loss(self, inputs, outputs, use_meta_mask: bool):
        batch_time_loss = compute_batch_time_loss(inputs, outputs)
        if use_meta_mask:
            if self.hparams.get('iterative_meta_mask', False):
                with torch.no_grad():
                    error = self.scenario.classifier_distance_torch(inputs, outputs)
                    meta_mask = error < self.hparams['meta_mask_threshold']
                    meta_mask = torch.logical_and(meta_mask[:, :-1], meta_mask[:, 1:]).float()
                    # self.log("model error", error.mean())
                    # self.log("iterative meta mask mean", meta_mask.mean())
                    meta_mask_padded = F.pad(meta_mask, [1, 0])
                if self.global_step > 10:  # skip the first few steps because training dynamics are weird...?
                    batch_time_loss = meta_mask_padded * batch_time_loss
            else:
                batch_time_loss = inputs['meta_mask'] * batch_time_loss
        batch_loss = batch_time_loss.sum(-1)

        if self.hparams.get('penalize_segment_length_error', False):
            initial_rope_segment_lengths = segment_lengths(inputs)
            pred_rope_segment_lengths = segment_lengths(outputs)
            pred_segment_length_loss = F.mse_loss(pred_rope_segment_lengths, initial_rope_segment_lengths)
            batch_loss = batch_time_loss + pred_segment_length_loss

        return batch_loss

    def compute_loss(self, inputs, outputs, use_meta_mask: bool):
        batch_loss = self.compute_batch_loss(inputs, outputs, use_meta_mask)
        return batch_loss.mean()

    def training_step(self, train_batch, batch_idx):
        outputs = self.forward(train_batch)
        use_meta_mask = self.hparams.get('use_meta_mask_train', False)
        loss = self.compute_loss(train_batch, outputs, use_meta_mask)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        val_udnn_outputs = self.forward(val_batch)
        use_meta_mask = self.hparams.get('use_meta_mask_val', False)
        val_loss = self.compute_loss(val_batch, val_udnn_outputs, use_meta_mask)
        self.log('val_loss', val_loss)

        model_error_batch = (val_batch['rope'] - val_udnn_outputs['rope']).norm(dim=-1).flatten()
        if self.val_model_errors is None:
            self.val_model_errors = model_error_batch
        else:
            self.val_model_errors = torch.cat([self.val_model_errors, model_error_batch])
        return val_loss

    def test_step(self, test_batch, batch_idx):
        test_udnn_outputs = self.forward(test_batch)
        test_loss = self.compute_loss(test_batch, test_udnn_outputs, use_meta_mask=False)
        self.log('test_loss', test_loss)

        model_error_batch = (test_batch['rope'] - test_udnn_outputs['rope']).norm(dim=-1).flatten()
        if self.test_model_errors is None:
            self.test_model_errors = model_error_batch
        else:
            self.test_model_errors = torch.cat([self.test_model_errors, model_error_batch])
        return test_loss

    def validation_epoch_end(self, _):
        data = self.val_model_errors.cpu().unsqueeze(-1).numpy().tolist()
        table = wandb.Table(data=data, columns=["model_errors"])
        wandb.log({'val_model_error': table})

        # reset all metrics
        self.val_model_errors = None

    def test_epoch_end(self, _):
        data = self.test_model_errors.cpu().unsqueeze(-1).numpy().tolist()
        table = wandb.Table(data=data, columns=["model_errors"])
        wandb.log({'test_model_error': table})

        # reset all metrics
        self.test_model_errors = None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def compute_batch_time_loss(inputs, outputs):
    loss_by_key = []
    for k, y_pred in outputs.items():
        y_true = inputs[k]
        # mean over time and state dim but not batch, not yet.
        loss = (y_true - y_pred).square().mean(-1)
        loss_by_key.append(loss)
    batch_time_loss = torch.stack(loss_by_key).mean(0)
    return batch_time_loss
