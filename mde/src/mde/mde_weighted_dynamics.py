from typing import Dict

import pytorch_lightning as pl
import torch
from torch.nn import Parameter

from link_bot_data.dataset_utils import add_predicted
from mde.mde_torch import MDE
from state_space_dynamics.meta_udnn import UDNN


class MDEWeightedDynamics(pl.LightningModule):

    def __init__(self, **hparams):
        super().__init__()

        self.udnn = UDNN(**hparams['udnn'])
        self.mde = MDE(**hparams['mde'])

        self.save_hyperparameters()

        self.scenario = self.udnn.scenario

        data_collection_params = self.hparams['dataset_hparams']['data_collection_params']
        self.env_keys = list(data_collection_params['env_description'].keys())
        self.action_keys = list(data_collection_params['action_description'].keys())
        self.state_keys = list(data_collection_params['state_description'].keys())
        self.state_metadata_keys = list(data_collection_params['state_metadata_description'].keys())

        self.register_parameter("learned_weight_k", Parameter(torch.tensor(10.0)))

        self.has_checked_training_mode = False

    def forward(self, inputs):
        return self.udnn.forward(inputs)

    def predict_dynamics_and_error(self, inputs: Dict[str, torch.Tensor]):
        # convert inputs to [b*44, 2, n] for time state inputs
        # we first generate all transitions, then flatten into batch dimension
        state_keys = self.state_keys + self.state_metadata_keys
        time_idx = inputs['time_idx']

        for k in state_keys:
            v = inputs[k]
            t = torch.arange(10)
            torch.stack([t[:-1], t[1:]], -1)
            indices = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]])
            v[0:], v[1:], ..., v[9:]
            v[..., indices, :].shape

        inputs_all_transitions = inputs
        inputs_all_transitions.update(states_all_transitions)
        inputs_all_transitions.update(actions_all_transitions)

        udnn_outputs = self.udnn.forward(inputs_all_transitions)
        mde_inputs = {}
        mde_inputs.update(inputs)
        mde_inputs.update({add_predicted(k): v for k, v in udnn_outputs.items()})
        # because the joint positions are not being changed, we can just copy these
        mde_inputs[add_predicted('joint_positions')] = inputs['joint_positions']
        mde_inputs['error'] = self.scenario.classifier_distance_torch(inputs, udnn_outputs)

        # currently we have:
        # rope [b, T, n]
        # predicted/rope [b, T, n]
        # env [b, w, h, c]
        # origin_point [b, 3]
        # res [b]

        # what we want:
        # rope [b, 2, n]
        # predicted/rope [b, 2, n]
        # env [b, w, h, c]
        # origin_point [b, 3]
        # res [b]

        # The MDE is written to work on transitions of length 2, but these are entire trajectories of length 10,
        # so we'd need to figure how to convert these into all possible transitions, then pass that to the MDE?
        mde_outputs = self.mde.forward(mde_inputs)

        return udnn_outputs, mde_outputs

    def compute_loss(self, inputs, predictions, mde_outputs):
        batch_dynamics_loss = self.udnn.compute_batch_loss(inputs, predictions, use_meta_mask=False)
        mde_loss = self.mde.compute_loss(inputs, mde_outputs)
        weights = torch.exp(-self.learned_weight_k * mde_outputs['error'])
        weighted_dynamics_loss = batch_dynamics_loss * weights
        loss = weighted_dynamics_loss + mde_loss
        return {
            'dynamics_loss': weighted_dynamics_loss,
            'mde_loss':      mde_loss,
            'loss':          loss
        }

    def training_step(self, train_batch: Dict[str, torch.Tensor], batch_idx):
        udnn_outputs, mde_outputs = self.predict_dynamics_and_error(train_batch)
        losses = self.compute_loss(train_batch, udnn_outputs, mde_outputs)
        for k, v in losses:
            self.log(f'train_{k}', v)
        return losses

    def validation_step(self, val_batch: Dict[str, torch.Tensor], batch_idx):
        udnN_outputs, mde_outputs = self.predict_dynamics_and_error(val_batch)
        losses = self.compute_loss(val_batch, udnN_outputs, mde_outputs)
        true_error = val_batch['error'][:, 1]
        true_error_thresholded = true_error < self.hparams.error_threshold
        pred_error_thresholded = mde_outputs < self.hparams.error_threshold
        for k, v in losses:
            self.log(f'val_{k}', v)
        self.val_accuracy(pred_error_thresholded, true_error_thresholded)  # updates the metric
        return losses

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.hparams.learning_rate,
                                weight_decay=self.hparams.get('weight_decay', 0))
