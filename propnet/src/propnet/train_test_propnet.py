#!/usr/bin/env python

import logging
import multiprocessing
import pathlib
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.filepath_tools import load_hjson
from moonshine.moonshine_utils import numpify
from moonshine.torch_utils import my_collate
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
                              num_workers=get_num_workers(batch_size),
                              collate_fn=my_collate)
    val_loader = None
    if len(val_dataset) > 0 and not no_validate:
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=get_num_workers(batch_size),
                                collate_fn=my_collate)

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
    early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    trainer = pl.Trainer(gpus=1,
                         weights_summary=None,
                         log_every_n_steps=1,
                         max_epochs=epochs,
                         callbacks=[best_val_ckpt_cb, latest_ckpt_cb, early_stopping])
    trainer.fit(model, train_loader, val_loader)


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

            pred_state_t = {}
            height_b_t = inputs['height'][b, t]
            pred_state_t['height'] = height_b_t
            pred_state_t['radius'] = inputs['radius'][b, t]
            num_objs = inputs['num_objs'][b, t, 0]
            pred_state_t['num_objs'] = [num_objs]
            for j in range(num_objs):
                pred_pos_b_t_2d = pred_pos[b, t, j + 1]
                pred_pos_b_t_3d = torch.cat([pred_pos_b_t_2d, height_b_t / 2])
                pred_vel_b_t_2d = pred_vel[b, t, j + 1]
                pred_vel_b_t_3d = torch.cat([pred_vel_b_t_2d, torch.zeros(1)])
                pred_state_t[f'obj{j}/position'] = torch.unsqueeze(pred_pos_b_t_3d, 0)
                # pred_state_t[f'obj{j}/linear_velocity'] = torch.unsqueeze(pred_vel_b_t_3d, 0)

            s.plot_state_rviz(pred_state_t, label='predicted', color='#0000ffaa')

            anim.step()


def get_num_workers(batch_size):
    return min(batch_size, multiprocessing.cpu_count())
    # return 0
