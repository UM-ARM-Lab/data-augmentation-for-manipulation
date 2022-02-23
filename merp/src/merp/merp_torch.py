import pytorch_lightning as pl
import torch

from link_bot_pycommon.get_scenario import get_scenario


class MERP(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        datset_params = hparams['dataset_hparams']
        data_collection_params = datset_params['data_collection_params']
        self.scenario = get_scenario(self.hparams.scenario, params=data_collection_params['scenario_params'])

    def forward(self, inputs):
        pass

    def compute_loss(self, inputs, outputs):
        return loss

    def training_step(self, train_batch, batch_idx):
        outputs = self.forward(train_batch)
        loss = self.compute_loss(train_batch, outputs)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        outputs = self.forward(val_batch)
        loss = self.compute_loss(val_batch, outputs)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
