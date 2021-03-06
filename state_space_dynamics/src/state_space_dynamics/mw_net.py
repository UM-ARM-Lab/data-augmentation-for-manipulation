import math
from collections import OrderedDict
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from colorama import Fore
from torch.nn.parameter import Parameter
from torchmeta.utils import gradient_update_parameters
from tqdm import tqdm

from moonshine.numpify import numpify
from moonshine.torch_and_tf_utils import remove_batch, add_batch
from moonshine.torchify import torchify
from state_space_dynamics.meta_udnn import UDNN


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


def model_error_to_unnormalized_weight(model_error_i, a, b, c):
    unnormalized_weight = a * (torch.exp(-b * model_error_i) - c)
    return unnormalized_weight


def solve_exponential_system():
    # solve system of exponential equations.
    # we solve for a,b,c in $$y=a(e^{-bx}-c)$$
    # given three data points (x, y)
    # this lets us map a chosen model_error (e.g. 0) to a give unnormalized weight (e.g. -10)
    # which after being normalized via the sigmoid gives us a the desired data weight (e.g. very small)
    from scipy.optimize import fsolve
    def _func(params):
        a, b, c = params
        x = np.array([0, 0.1, 0.5])
        y = np.array([10, 0, -10])
        cost = (a * (np.exp(-b * x) - c) - y) ** 2
        return cost

    a, b, c = fsolve(_func, [10, 5, 1])
    return a, b, c


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

        self.testing = False
        self.automatic_optimization = False
        self.params_states = {}

    def forward(self, inputs):
        return self.udnn.forward(inputs)

    def init_data_weights_from_model_error(self, train_dataset):
        if self.hparams.init_from_model_error:
            print(Fore.GREEN + "Initializing data weights" + Fore.RESET)
            # TODO: do this elsewhere, refactor!
            # compute the model error using the initial model on the initial dataset
            a, b, c = solve_exponential_system()
            for example_i in tqdm(train_dataset):
                train_example_i = example_i['train']
                i = train_example_i['example_idx']
                outputs = numpify(remove_batch(self.udnn(add_batch(torchify(train_example_i)))))
                model_error_i_time = self.udnn.scenario.classifier_distance(train_example_i, outputs)
                model_error_i = torch.tensor(model_error_i_time.mean())
                init_unnormalized_weight = model_error_to_unnormalized_weight(model_error_i, a, b, c)
                with torch.no_grad():
                    self.sample_weights[i] = init_unnormalized_weight

    def validation_step(self, inputs, batch_idx):
        meta_train_batch = inputs['meta_train']
        meta_train_udnn_outputs = self.udnn(meta_train_batch)
        use_meta_mask = not self.testing
        meta_train_udnn_loss = self.udnn.compute_loss(meta_train_batch, meta_train_udnn_outputs,
                                                      use_meta_mask=use_meta_mask)
        self.log('val_loss', meta_train_udnn_loss)

    def training_step(self, inputs, batch_idx):
        # evaluate the validation loss
        meta_train_batch = inputs['meta_train']
        meta_train_udnn_outputs = self.udnn(meta_train_batch)
        meta_train_udnn_loss = self.udnn.compute_loss(meta_train_batch, meta_train_udnn_outputs, use_meta_mask=True)
        self.log('val_loss', meta_train_udnn_loss)

        train_data_weights = self.sample_weights.detach().cpu()[self.hparams['train_example_indices']]
        wandb.log({'unnormalized weights': wandb.Histogram(train_data_weights)})

        # start of actually meta-learning
        self.sample_weights.grad = None  # zero grad, very important!
        self.udnn.zero_grad()

        data_weight_opt = self.optimizers()

        train_batch = inputs['train']

        udnn_outputs = self.udnn(train_batch)
        udnn_loss = self.udnn.compute_batch_loss(train_batch, udnn_outputs, use_meta_mask=False)
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
        meta_train_udnn_loss = self.udnn.compute_loss(meta_train_batch, meta_train_udnn_outputs, use_meta_mask=True)
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
