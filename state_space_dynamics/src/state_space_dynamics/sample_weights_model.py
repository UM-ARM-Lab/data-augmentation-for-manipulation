from typing import Optional

import pytorch_lightning as pl
import torch
import wandb
from torch.nn import Parameter


class SampleWeightsModel(pl.LightningModule):
    def __init__(self, train_dataset: Optional, model: pl.LightningModule, **hparams):
        super().__init__()
        self.save_hyperparameters(ignore=['train_dataset', 'model'])

        self.model = model

        if train_dataset is not None:
            max_example_idx = max([e['example_idx'] for e in train_dataset])
            initial_sample_weights = torch.ones(max_example_idx + 1)
            self.register_parameter("sample_weights", Parameter(initial_sample_weights))

    def forward(self, inputs):
        return self.model.forward(inputs)

    def training_step(self, train_batch, batch_idx):
        outputs = self.model.forward(train_batch)
        batch_loss = self.model.compute_batch_loss(train_batch, outputs)
        batch_indices = train_batch['example_idx']
        sample_weights_for_batch = torch.take_along_dim(self.sample_weights, batch_indices, dim=0)
        sample_weights_for_batch = torch.clip(sample_weights_for_batch, 0, 1)
        loss = sample_weights_for_batch @ batch_loss - sample_weights_for_batch.sum()
        self.log('weighted_train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        outputs = self.model.forward(val_batch)
        val_loss = self.model.compute_loss(val_batch, outputs)
        self.log('val_loss', val_loss)
        wandb.log({
            'weights': wandb.Histogram(self.sample_weights),
        })
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
