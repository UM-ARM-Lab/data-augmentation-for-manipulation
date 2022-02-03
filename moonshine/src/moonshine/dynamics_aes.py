from moonshine.ae import MyAE
from moonshine.make_gifs_callbacks import log_reconstruction_gifs
from moonshine.vae import MyVAE


# noinspection PyAbstractClass
class DynamicsVAE(MyVAE):

    def __init__(self, hparams, scenario=None):
        super().__init__(hparams, scenario=scenario)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.current_epoch % 50 == 0:
            log_reconstruction_gifs(self.scenario, batch, outputs, 'train')

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            log_reconstruction_gifs(self.scenario, batch, outputs, 'val')


# noinspection PyAbstractClass
class DynamicsAE(MyAE):

    def __init__(self, hparams, scenario=None):
        super().__init__(hparams, scenario=scenario)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.current_epoch % 50 == 0:
            log_reconstruction_gifs(self.scenario, batch, outputs, 'train')

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            log_reconstruction_gifs(self.scenario, batch, outputs, 'val')
