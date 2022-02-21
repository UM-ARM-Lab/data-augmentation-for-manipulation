import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from link_bot_pycommon.load_wandb_model import load_model_artifact
from moonshine.torch_utils import vector_to_dict, sequence_of_dicts_to_dict_of_tensors, loss_on_dicts
from state_space_dynamics.udnn_torch import UDNN


class BaseResUDNN(pl.LightningModule):
    def __init__(self, hparams, udnn: UDNN):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.udnn = udnn
        self.scenario = self.udnn.scenario
        self.state_description = self.udnn.state_description
        in_size = self.udnn.total_state_dim + self.udnn.total_action_dim
        fc_layer_size = None

        layers = []
        for fc_layer_size in self.hparams.fc_layer_sizes:
            layers.append(torch.nn.Linear(in_size, fc_layer_size))
            layers.append(torch.nn.ReLU())
            in_size = fc_layer_size
        layers.append(torch.nn.Linear(fc_layer_size, self.udnn.total_state_dim))

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
        return pred_states_dict

    def one_step_forward(self, action_t, s_t):
        s_t_local = self.scenario.put_state_local_frame_torch(s_t)
        local_action_t = self.scenario.put_action_local_frame(s_t, action_t)

        states_and_actions = list(s_t_local.values()) + list(local_action_t.values())
        z_t = torch.concat(states_and_actions, -1)
        z_t = self.mlp(z_t)
        res_s_t_plus_1 = vector_to_dict(self.state_description, z_t, self.device)

        base_s_t_plus_1 = self.udnn.one_step_forward(action_t, s_t)

        s_t_plus_1 = self.scenario.integrate_dynamics(base_s_t_plus_1, res_s_t_plus_1)
        return s_t_plus_1

    def compute_loss(self, inputs, outputs):
        loss_by_key = []
        for k, y_pred in outputs.items():
            y_true = inputs[k]
            # mean over time and state dim but not batch, not yet.
            loss = (y_true - y_pred).square().mean(dim=-1).mean(dim=-1)
            loss_by_key.append(loss)
        batch_loss = torch.stack(loss_by_key).mean(dim=0)

        if 'weights' in inputs:
            weights = inputs['weights']
        else:
            weights = torch.ones_like(batch_loss).to(self.device)
        loss = batch_loss @ weights

        return loss

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


class ResUDNN(BaseResUDNN):
    def __init__(self, hparams):
        udnn = load_model_artifact(hparams['udnn_checkpoint'], UDNN, project='udnn', version='best', user='armlab')
        super().__init__(hparams, udnn)
