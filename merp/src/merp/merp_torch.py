from typing import Dict

import pytorch_lightning as pl
import torch
from torch import nn


# duplicated and tweak for compatibility with torch jit
def add_predicted(feature_name: str):
    return "predicted/" + feature_name


class MERP(pl.LightningModule):

    def __init__(self, hparams, scenario):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.scenario = scenario

        self.mlp = nn.Sequential(nn.Linear(2 * 25 * 3, 1))

    def forward(self, inputs: Dict[str, torch.Tensor]):
        return self.mlp(inputs[add_predicted('rope')].reshape(-1, 2 * 25 * 3))

    def compute_loss(self, inputs: Dict[str, torch.Tensor], outputs):
        return (outputs - inputs[add_predicted('rope')].reshape(-1, 2 * 25 * 3).sum(-1, keepdims=True)).square().sum()

    def training_step(self, train_batch: Dict[str, torch.Tensor], batch_idx):
        outputs = self.forward(train_batch)
        loss = self.compute_loss(train_batch, outputs)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch: Dict[str, torch.Tensor], batch_idx):
        outputs = self.forward(val_batch)
        loss = self.compute_loss(val_batch, outputs)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
