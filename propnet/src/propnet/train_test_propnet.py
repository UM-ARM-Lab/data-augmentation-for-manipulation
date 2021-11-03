#!/usr/bin/env python
import logging
import multiprocessing
import pathlib
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from moonshine.filepath_tools import load_hjson
from propnet.models import PropNet
from propnet.torch_dynamics_dataset import TorchDynamicsDataset


def train_main(dataset_dir: pathlib.Path,
               model_params: pathlib.Path,
               log: str,
               batch_size: int,
               epochs: int,
               seed: int,
               checkpoint: Optional[pathlib.Path] = None,
               take: Optional[int] = None,
               no_validate: bool = False,
               **kwargs):
    train_dataset = TorchDynamicsDataset(dataset_dir, mode='train')
    val_dataset = TorchDynamicsDataset(dataset_dir, mode='val')
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=get_num_workers(batch_size))
    val_loader = None
    if len(val_dataset) > 0 and not no_validate:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=get_num_workers(batch_size))

    if take:
        train_loader = train_loader[:take]
        val_loader = val_loader[:take]

    if checkpoint is None:
        model_params = load_hjson(model_params)
        model_params['num_objects'] = train_dataset.params['data_collection_params']['num_objs'] + 1
        model_params['scenario'] = train_dataset.params['scenario']
        model = PropNet(hparams=model_params)
    else:
        model = PropNet.load_from_checkpoint(checkpoint.as_posix())

    # training
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    best_val_ckpt_cb = pl.callbacks.ModelCheckpoint(monitor="val_loss",
                                                    filename="best")
    latest_ckpt_cb = pl.callbacks.ModelCheckpoint(filename='latest')
    trainer = pl.Trainer(gpus=1,
                         weights_summary=None,
                         log_every_n_steps=1,
                         max_epochs=epochs,
                         callbacks=[best_val_ckpt_cb, latest_ckpt_cb])
    trainer.fit(model, train_loader, val_loader)


def viz_main(dataset_dir: pathlib.Path, checkpoint: pathlib.Path, mode: str, **kwargs):
    dataset = TorchDynamicsDataset(dataset_dir, mode)
    s = dataset.get_scenario()

    loader = DataLoader(dataset)

    model = PropNet.load_from_checkpoint(checkpoint.as_posix())

    for i, inputs in enumerate(loader):
        outputs = model(inputs)

        s.plot_state_rviz(inputs, 'predicted', color='r')
        s.plot_state_rviz(inputs, 'actual', color='b')


def get_num_workers(batch_size):
    return min(batch_size, multiprocessing.cpu_count())
    # return 0
