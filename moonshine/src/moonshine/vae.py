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
            nn.Linear(self.hparams.hidden_dim, self.hparams.latent_dim),
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

        self.fc_mu = nn.Linear(self.hparams.latent_dim, self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.hparams.latent_dim, self.hparams.latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(-1)

    def kl_divergence(self, z, mu, std):
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def forward(self, example):
        x = self.scenario.example_dict_to_flat_vector(example)
        _, x_reconstruction = self.flat_vector_forward(x)
        reconstruction_dict = self.scenario.flat_vector_to_example_dict(example, x_reconstruction)
        return reconstruction_dict

    def flat_vector_forward(self, x):
        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        return elbo, x_hat

    def training_step(self, train_batch, batch_idx):
        x = self.scenario.example_dict_to_flat_vector(train_batch)
        loss, x_reconstruction = self.flat_vector_forward(x)

        self.log('train_loss', loss)

        return {
            'loss':             loss,
            'x_reconstruction': x_reconstruction.detach(),
        }

    def validation_step(self, val_batch, batch_idx):
        x = self.scenario.example_dict_to_flat_vector(val_batch)
        loss, x_reconstruction = self.flat_vector_forward(x)

        self.log('val_loss', loss)

        return {
            'loss':             loss,
            'x_reconstruction': x_reconstruction.detach(),
        }
