import threading

import pytorch_lightning as pl


class MyModel(pl.LightningModule):

    def __init__(self, non_serializable, hparams):
        super().__init__()
        self.non_serializable = non_serializable
        self.save_hyperparameters(hparams)


if __name__ == '__main__':
    m = MyModel(threading.RLock(), {})
