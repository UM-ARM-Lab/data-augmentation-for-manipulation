import logging
import tempfile

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb
from torch import nn

logger = logging.getLogger(__file__)

video_format = 'gif'
fps = 60
train_gif_log_idx = 0


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
        reconstruction_dict = self.scenario.flat_vector_to_example_dict(train_batch, x_reconstruction)
        loss = self.loss(x, x_reconstruction)
        self.log('train_loss', loss)

        # HACK!!!
        global train_gif_log_idx
        if train_gif_log_idx % 50 == 0:
            input_anim = self.scenario.example_to_gif(train_batch)
            input_gif_filename = tempfile.mktemp(suffix=f'.{video_format}')
            input_anim.save(input_gif_filename, writer='imagemagick', fps=fps)
            plt.close(input_anim._fig)
            reconstruction_anim = self.scenario.example_to_gif(reconstruction_dict)
            reconstruction_gif_filename = tempfile.mktemp(suffix=f'.{video_format}')
            reconstruction_anim.save(reconstruction_gif_filename, writer='imagemagick', fps=fps)
            plt.close(reconstruction_anim._fig)
            wandb.log({
                'train_input_gif':          wandb.Video(input_gif_filename, fps=fps, format=video_format),
                'train_reconstruction_gif': wandb.Video(reconstruction_gif_filename, fps=fps, format=video_format),
            })
        train_gif_log_idx += 1
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = self.scenario.example_dict_to_flat_vector(val_batch)
        x_reconstruction = self.flat_vector_forward(x)
        reconstruction_dict = self.scenario.flat_vector_to_example_dict(val_batch, x_reconstruction)
        loss = self.loss(x, x_reconstruction)
        self.log('val_loss', loss)

        input_anim = self.scenario.example_to_gif(val_batch)
        input_gif_filename = tempfile.mktemp(suffix=f'.{video_format}')
        input_anim.save(input_gif_filename, writer='imagemagick', fps=fps)
        plt.close(input_anim._fig)
        reconstruction_anim = self.scenario.example_to_gif(reconstruction_dict)
        reconstruction_gif_filename = tempfile.mktemp(suffix=f'.{video_format}')
        reconstruction_anim.save(reconstruction_gif_filename, writer='imagemagick', fps=fps)
        plt.close(reconstruction_anim._fig)

        wandb.log({
            'val_input_gif':          wandb.Video(input_gif_filename, fps=fps, format=video_format),
            'val_reconstruction_gif': wandb.Video(reconstruction_gif_filename, fps=fps, format=video_format),
        })