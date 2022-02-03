import logging

import pytorch_lightning as pl
import torch
from torch import nn

logger = logging.getLogger(__file__)


def reparametrize(mu, log_var):
    # Reparametrization Trick to allow gradients to backpropagate from the stochastic part of the model
    sigma = torch.exp(0.5 * log_var)
    z = torch.randn_like(sigma)
    return mu + sigma * z


# noinspection PyAbstractClass
class MyVAE(pl.LightningModule):

    def __init__(self, hparams, scenario=None):
        super().__init__()
        self.scenario = scenario
        self.save_hyperparameters(hparams)
        self.encoder = nn.Sequential(
            nn.Linear(self.hparams.input_dim, self.hparams.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hparams.hidden_dim, self.hparams.latent_dim * 2),  # times 2 because mean and variance
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hparams.latent_dim, self.hparams.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hparams.hidden_dim, self.hparams.input_dim),
        )

    def flat_vector_forward(self, x):
        h = self.encoder(x)
        mu = h[..., :int(self.hparams.latent_dim)]
        log_var = h[..., int(self.hparams.latent_dim):]
        hidden = reparametrize(mu, log_var)
        output = self.decoder(hidden)
        return output

    def forward(self, example):
        x = self.scenario.example_dict_to_flat_vector(example)
        x_reconstruction = self.flat_vector_forward(x)
        reconstruction_dict = self.scenario.flat_vector_to_example_dict(example, x_reconstruction)
        return reconstruction_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def loss(self, batch, reconstruction):
        return torch.norm(batch - reconstruction)

    def training_step(self, train_batch, batch_idx):
        x = self.scenario.example_dict_to_flat_vector(train_batch)
        x_reconstruction = self.flat_vector_forward(x)
        loss = self.loss(x, x_reconstruction)
        self.log('train_loss', loss)

        return {
            'loss':             loss,
            'x_reconstruction': x_reconstruction.detach(),
        }

    def validation_step(self, val_batch, batch_idx):
        x = self.scenario.example_dict_to_flat_vector(val_batch)
        x_reconstruction = self.flat_vector_forward(x)
        loss = self.loss(x, x_reconstruction)
        self.log('val_loss', loss)

        return {
            'loss':             loss,
            'x_reconstruction': x_reconstruction.detach(),
        }
