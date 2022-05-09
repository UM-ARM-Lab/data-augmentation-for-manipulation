import math
import pathlib
from collections import OrderedDict
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import wandb
from colorama import Fore
from torch.nn.parameter import Parameter
from torchmeta.modules import MetaModule, MetaLinear, MetaSequential
from torchmeta.utils import gradient_update_parameters
from tqdm import tqdm

from link_bot_pycommon.get_scenario import get_scenario
from moonshine.numpify import numpify
from moonshine.robot_points_torch import RobotVoxelgridInfo
from moonshine.torch_and_tf_utils import remove_batch, add_batch
from moonshine.torch_utils import sequence_of_dicts_to_dict_of_tensors, vector_to_dict
from moonshine.torchify import torchify
from state_space_dynamics.heuristic_data_weights import heuristic_weight_func
from state_space_dynamics.udnn_torch import mask_after_first_0, compute_batch_time_loss


def gradient_update_parameters(model, loss, step_size):
    params = OrderedDict(model.meta_named_parameters())
    grads = torch.autograd.grad(loss, params.values(), create_graph=True)
    updated_params = OrderedDict()

    for (name, param), grad in zip(params.items(), grads):
        updated_params[name] = param - step_size * grad

    return updated_params


def adam_update(model, loss, step_size, params_states, betas=(0.9, 0.999), eps=1e-8):
    params = OrderedDict(model.meta_named_parameters())
    grads = torch.autograd.grad(loss, params.values(), create_graph=True)
    updated_params = OrderedDict()

    b1, b2 = betas
    for (name, param), grad in zip(params.items(), grads):
        if param not in params_states:
            params_states[param] = {}
        param_state = params_states[param]

        # State initialization
        if len(param_state) == 0:
            param_state['step'] = 0
            # Momentum (Exponential MA of gradients)
            param_state['exp_avg'] = torch.zeros_like(param)
            # RMS Prop componenet. (Exponential MA of squared gradients). Denominator.
            param_state['exp_avg_sq'] = torch.zeros_like(param)

        exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']

        param_state['step'] += 1

        # Momentum
        exp_avg = torch.mul(exp_avg, b1) + (1 - b1) * grad
        # RMS
        # I added this +eps for when grad=0, the 1/sqrt() op gives a NaN gradient for d_updated_params/d_grad
        exp_avg_sq = torch.mul(exp_avg_sq, b2) + (1 - b2) * (grad * grad) + eps

        denom = exp_avg_sq.sqrt() + eps

        bias_correction1 = 1 / (1 - b1 ** param_state['step'])
        bias_correction2 = 1 / (1 - b2 ** param_state['step'])

        adapted_learning_rate = step_size * bias_correction1 / math.sqrt(bias_correction2)

        updated_params[name] = param - adapted_learning_rate * exp_avg / denom

    return updated_params


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
        self.max_step_size = self.data_collection_params['max_step_size']

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

    def validation_step(self, val_batch, batch_idx):
        val_udnn_outputs = self.forward(val_batch)
        val_loss = self.compute_loss(val_batch, val_udnn_outputs)
        self.log('val_loss', val_loss)


class MWNet(pl.LightningModule):
    def __init__(self, train_dataset: Optional, **hparams):
        super().__init__()

        if train_dataset is not None:
            train_example_indices = [e['train']['example_idx'] for e in train_dataset]
            max_example_idx = max(train_example_indices)
            self.hparams['max_example_idx'] = max_example_idx
            self.hparams['train_example_indices'] = train_example_indices
        else:
            max_example_idx = hparams['max_example_idx']

        self.save_hyperparameters(ignore=['train_dataset'])

        self.udnn = UDNN(**self.hparams)

        initial_sample_weights = torch.ones(max_example_idx + 1) * self.hparams.data_weight_init
        self.register_parameter("sample_weights", Parameter(initial_sample_weights))

        self.state_keys = self.udnn.state_keys
        self.state_metadata_keys = self.udnn.state_metadata_keys

        self.automatic_optimization = False
        self.params_states = {}

    def forward(self, inputs):
        return self.udnn.forward(inputs)

    def init_data_weights_from_model_error(self, train_dataset):
        if self.hparams.init_from_model_error:
            print(Fore.GREEN + "Initializing data weights" + Fore.RESET)
            # TODO: do this elsewhere, refactor!
            # compute the model error using the initial model on the initial dataset
            for example_i in tqdm(train_dataset):
                train_example_i = example_i['train']
                i = train_example_i['example_idx']
                model_error_i_dict = remove_batch(self.udnn(add_batch(torchify(train_example_i))))
                model_error_i = 0
                for v in model_error_i_dict.values():
                    model_error_i += v.mean()
                with torch.no_grad():
                    self.sample_weights[i] = model_error_i

    def init_data_weights_from_heuristic(self, train_dataset):
        print(Fore.GREEN + "Initializing data weights" + Fore.RESET)
        hparams = {
            'heuristic_weighting': True,  # don't change this, it's just metadata
            'env_inflation':       0.9,
            'check_robot':         True,
            'robot_inflation':     0.6,
            'max_rope_length':     0.774,
            'check_length':        False,
        }
        robot_points_path = pathlib.Path("robot_points_data/val_high_res/robot_points.pkl")
        robot_info = RobotVoxelgridInfo('joint_positions', robot_points_path)
        for example_i in tqdm(train_dataset):
            heuristic_weight_func(self.scenario, example_i, hparams, robot_info)

    def validation_step(self, inputs, batch_idx):
        meta_train_batch = inputs['meta_train']
        meta_train_udnn_outputs = self.udnn(meta_train_batch)
        meta_train_udnn_loss = self.udnn.compute_loss(meta_train_batch, meta_train_udnn_outputs)
        self.log('val_loss', meta_train_udnn_loss)

    def training_step(self, inputs, batch_idx):
        # evaluate the validation loss
        meta_train_batch = inputs['meta_train']
        meta_train_udnn_outputs = self.udnn(meta_train_batch)
        meta_train_udnn_loss = self.udnn.compute_loss(meta_train_batch, meta_train_udnn_outputs)
        self.log('val_loss', meta_train_udnn_loss)

        train_data_weights = self.sample_weights.detach().cpu()[self.hparams['train_example_indices']]
        wandb.log({'unnormalized weights': wandb.Histogram(train_data_weights)})

        self.sample_weights.grad = None  # zero grad, very important!
        self.udnn.zero_grad()

        data_weight_opt = self.optimizers()

        train_batch = inputs['train']

        udnn_outputs = self.udnn(train_batch)
        udnn_loss = self.udnn.compute_batch_loss(train_batch, udnn_outputs)
        batch_indices = train_batch['example_idx']
        weights = torch.take_along_dim(self.sample_weights, batch_indices, dim=0)
        positive_weights = torch.sigmoid(weights)
        udnn_loss_weighted = torch.sum(udnn_loss * positive_weights) / udnn_loss.nelement()  # inner loss

        self.log('train_loss', udnn_loss_weighted)

        # compute the update for udnn and get the updated params
        if self.hparams.get('adam', False):
            updated_params = adam_update(self.udnn, udnn_loss_weighted, step_size=self.hparams.udnn_inner_learning_rate,
                                         params_states=self.params_states)
        else:
            updated_params = gradient_update_parameters(self.udnn,
                                                        udnn_loss_weighted,
                                                        step_size=self.hparams.udnn_inner_learning_rate,
                                                        first_order=False)

        meta_train_batch = inputs['meta_train']
        meta_train_udnn_outputs = self.udnn(meta_train_batch, params=updated_params)
        meta_train_udnn_loss = self.udnn.compute_loss(meta_train_batch, meta_train_udnn_outputs)
        meta_train_udnn_loss.backward()
        data_weight_opt.step()  # updates data weights

        self.udnn.load_state_dict(updated_params)

    def configure_optimizers(self):
        data_weight_opt = torch.optim.SGD([self.sample_weights], lr=self.hparams.weight_learning_rate)

        def _clip(grad):
            return torch.clamp(grad, -self.hparams.grad_clip_value, self.hparams.grad_clip_value)

        self.sample_weights.register_hook(_clip)
        for p in self.udnn.parameters():
            p.register_hook(_clip)

        return data_weight_opt
