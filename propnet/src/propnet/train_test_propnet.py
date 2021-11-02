#!/usr/bin/env python
import pathlib
from typing import List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from moonshine.filepath_tools import load_hjson
from propnet.models import PropNet
from propnet.torch_dynamics_dataset import TorchDynamicsDataset


def train_main(dataset_dirs: List[pathlib.Path],
               model_params: pathlib.Path,
               log: str,
               batch_size: int,
               epochs: int,
               seed: int,
               checkpoint: Optional[pathlib.Path] = None,
               take: Optional[int] = None,
               no_validate: bool = False,
               trials_directory: Optional[pathlib.Path] = pathlib.Path("./trials").absolute(),
               **kwargs):
    model_params = load_hjson(model_params)

    assert len(dataset_dirs) == 1
    train_dataset = TorchDynamicsDataset(dataset_dirs[0], mode='train')
    val_dataset = TorchDynamicsDataset(dataset_dirs[0], mode='val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = None
    if len(val_dataset) > 0 and not no_validate:
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

    if take:
        train_loader = train_loader[:take]
        val_loader = val_loader[:take]

    model_params['num_objects'] = train_dataset.params['data_collection_params']['num_blocks'] + 1  # +1 for robot
    model = PropNet(params=model_params, scenario=train_dataset.get_scenario())

    # training
    trainer = pl.Trainer(gpus=1, weights_summary=None, log_every_n_steps=1, max_epochs=epochs)
    trainer.fit(model, train_loader, val_loader)
