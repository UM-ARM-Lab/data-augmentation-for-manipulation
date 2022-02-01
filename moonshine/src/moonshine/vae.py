import pytorch_lightning as pl
import torch
from torch import nn


# noinspection PyAbstractClass
class MyVAE(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.encoder = nn.Sequential(
            nn.Linear(self.hparams.input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.hparams.latent_dim * 2),  # times 2 because mean and variance
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hparams.latent_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.hparams.input_dim),
            nn.LeakyReLU(),
        )

    def reparametrize(self, mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the stochastic part of the model
        sigma = torch.exp(0.5 * log_var)
        z = torch.randn_like(sigma)
        return mu + sigma * z

    def forward(self, batch):
        h = self.encoder(batch)
        mu = h[..., :int(self.hparams.latent_dim)]
        log_var = h[..., int(self.hparams.latent_dim):]
        hidden = self.reparametrize(mu, log_var)
        output = self.decoder(hidden)
        return mu, log_var, output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def loss(self, batch, reconstruction):
        return torch.norm(batch - reconstruction)

    def training_step(self, train_batch, batch_idx):
        _, _, reconstruction = self.forward(train_batch)
        loss = self.loss(train_batch, reconstruction)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        _, _, reconstruction = self.forward(val_batch)
        loss = self.loss(val_batch, reconstruction)
        self.log('val_loss', loss)
