#!/usr/bin/env python

import logging
import multiprocessing
import pathlib
from datetime import datetime
from typing import Optional

import git
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.filepath_tools import load_hjson
from moonshine.moonshine_utils import numpify
from moonshine.torch_utils import my_collate
from propnet.propnet_models import PropNet
from propnet.torch_dynamics_dataset import TorchDynamicsDataset, remove_keys


def train_main(dataset_dir: pathlib.Path,
               model_params: pathlib.Path,
               batch_size: int,
               epochs: int,
               seed: int,
               checkpoint: Optional[pathlib.Path] = None,
               take: Optional[int] = None,
               no_validate: bool = False,
               **kwargs):
    pl.seed_everything(seed, workers=True)

    transform = transforms.Compose([
        remove_keys('filename', 'full_filename', 'joint_names', 'metadata'),
    ])

    train_dataset = TorchDynamicsDataset(dataset_dir, mode='train',
                                         transform=transform)
    val_dataset = TorchDynamicsDataset(dataset_dir, mode='val',
                                       transform=transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=get_num_workers(batch_size))
    val_loader = None
    if len(val_dataset) > 0 and not no_validate:
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=get_num_workers(batch_size))

    if take:
        train_loader = train_loader[:take]
        val_loader = val_loader[:take]

    if checkpoint is None:
        model_params = load_hjson(model_params)
        model_params['num_objects'] = train_dataset.params['data_collection_params']['num_objs'] + 1
        model_params['scenario'] = train_dataset.params['scenario']
        # add some extra useful info here
        stamp = "{:%B_%d_%H-%M-%S}".format(datetime.now())
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha[:10]
        model_params['sha'] = sha
        model_params['start-train-time'] = stamp
        model_params['dataset_dir'] = dataset_dir.as_posix()
        model_params['batch_size'] = batch_size
        model_params['seed'] = seed
        model_params['epochs'] = epochs
        model_params['take'] = take
        model_params['checkpoint'] = checkpoint
        model_params['no_validate'] = no_validate
        model = PropNet(hparams=model_params)
        ckpt_path = None
    else:
        ckpt_path = checkpoint.as_posix()
        model = PropNet.load_from_checkpoint(ckpt_path)

    # training
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    best_val_ckpt_cb = pl.callbacks.ModelCheckpoint(monitor="val_loss",
                                                    filename="best-{epoch:02d}-{val_loss:.6f}")
    latest_ckpt_cb = pl.callbacks.ModelCheckpoint(filename='latest-{epoch:02d}', save_on_train_epoch_end=True)
    early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", patience=50)
    callbacks = [
        best_val_ckpt_cb,
        latest_ckpt_cb,
        # early_stopping
    ]
    trainer = pl.Trainer(gpus=1,
                         enable_model_summary=False,
                         log_every_n_steps=1,
                         max_epochs=epochs,
                         callbacks=callbacks)
    trainer.fit(model,
                train_loader,
                val_dataloaders=val_loader,
                ckpt_path=ckpt_path)


def viz_main(dataset_dir: pathlib.Path, checkpoint: pathlib.Path, mode: str, **kwargs):
    dataset = TorchDynamicsDataset(dataset_dir, mode)
    s = dataset.get_scenario()

    loader = DataLoader(dataset, collate_fn=my_collate)

    model = PropNet.load_from_checkpoint(checkpoint.as_posix())

    for i, inputs in enumerate(loader):
        gt_vel, gt_pos, pred_vel, pred_pos = model(inputs)

        n_time_steps = inputs['time_idx'].shape[1]
        b = 0
        anim = RvizAnimationController(n_time_steps=n_time_steps)

        while not anim.done:
            t = anim.t()
            # FIXME: this is scenario specific!!!
            state_t = {}
            for k in dataset.state_keys:
                if k in inputs:
                    state_t[k] = numpify(inputs[k][b, t])

            s.plot_state_rviz(state_t, label='actual', color='#ff0000aa')

            pred_state_t = s.propnet_outputs_to_state(inputs=inputs, pred_vel=pred_vel, pred_pos=pred_pos, b=b, t=t)

            s.plot_state_rviz(pred_state_t, label='predicted', color='#0000ffaa')

            anim.step()


def get_num_workers(batch_size):
    return min(batch_size, multiprocessing.cpu_count())
    # return 0
