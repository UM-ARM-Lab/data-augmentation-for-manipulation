from typing import Dict

import pytorch_lightning as pl
import torch
from torch import nn

from state_space_dynamics.udnn_torch import UDNN


class VNet(pl.LightningModule):

    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        datset_params = self.hparams['dataset_hparams']
        data_collection_params = datset_params['data_collection_params']
        self.dataset_state_description: Dict = data_collection_params['state_description']
        self.state_description = {k: self.dataset_state_description[k] for k in self.hparams.state_keys}
        self.total_state_dim = sum([self.dataset_state_description[k] for k in self.hparams.state_keys])
        in_size = self.total_state_dim

        h = self.hparams['h']
        self.mlp = nn.Sequential(
            nn.Linear(in_size, h),
            nn.LeakyReLU(),
            nn.Linear(h, h),
            nn.LeakyReLU(),
            nn.Linear(h, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x)


class MWNet(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        self.udnn = UDNN(**hparams)
        self.vnet = VNet(**hparams['vnet'])
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
        new_state_dict = {}
        for (param_name, param_value), grad in zip(self.meta_udnn.named_parameters(), grads):
            new_state_dict[param_name] = param_value - self.hparams.learning_rate * grad
        self.meta_udnn.load_state_dict(new_state_dict)

        # now use the meta data batch, pass that through the meta_udnn, compute loss, then backprop to update VNet
        meta_train_batch = inputs['meta_train']
        meta_udnn_meta_outputs = self.meta_udnn(meta_train_batch)
        meta_loss_meta_batch = self.meta_udnn.compute_loss(meta_train_batch, meta_udnn_meta_outputs)
        optimizer_vnet.zero_grad()
        meta_loss_meta_batch.backward()
        optimizer_vnet.step()

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
