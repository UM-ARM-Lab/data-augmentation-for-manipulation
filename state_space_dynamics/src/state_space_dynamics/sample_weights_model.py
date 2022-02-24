from typing import Optional

import pytorch_lightning as pl
import torch
from torch.nn import Parameter

from link_bot_pycommon.load_wandb_model import load_model_artifact
from state_space_dynamics.udnn_torch import UDNN


class SampleWeightsModel(pl.LightningModule):
    def __init__(self, train_dataset: Optional, model: pl.LightningModule, **hparams):
        super().__init__()
        self.save_hyperparameters()

        self.model = model

        if train_dataset is not None:
            initial_sample_weights = torch.zeros(n_training_examples)
            self.register_parameter("sample_weights", Parameter(initial_sample_weights))

    def forward(self, inputs):
        return self.model.forward(inputs)

    def training_step(self, train_batch, batch_idx):
        unweighted_loss = self.model.training_step(train_batch, batch_idx)
        loss = self.sample_weights @ unweighted_loss
        self.log('weighted_train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        return self.model.validation_step(val_batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class ResUDNN(BaseResUDNN):
    def __init__(self, hparams):
        udnn = load_model_artifact(hparams['udnn_checkpoint'], UDNN, project='udnn', version='best', user='armlab')
        super().__init__(hparams, udnn)
