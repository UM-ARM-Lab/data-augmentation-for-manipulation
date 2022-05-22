#!/usr/bin/env python

import pathlib
from datetime import datetime
from typing import Optional

import git
import numpy as np
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from wandb.util import generate_id

from link_bot_data.visualization import init_viz_env, viz_pred_actual_t
from link_bot_data.wandb_datasets import get_dataset_with_version
from link_bot_pycommon.load_wandb_model import load_model_artifact, model_artifact_path
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.filepath_tools import load_hjson
from moonshine.moonshine_utils import get_num_workers
from moonshine.my_pl_callbacks import HeartbeatCallback
from moonshine.torch_and_tf_utils import add_batch, remove_batch
from moonshine.torch_datasets_utils import take_subset, dataset_skip, my_collate, repeat_dataset
from moonshine.torchify import torchify
from state_space_dynamics.meta_udnn import UDNN
from state_space_dynamics.mw_net import MWNet
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset, remove_keys

PROJECT = 'udnn'


def load_udnn_model_wrapper(checkpoint, with_joint_positions=False):
    try:
        model = load_model_artifact(checkpoint, UDNN, 'udnn', version='best', user='armlab',
                                    with_joint_positions=with_joint_positions)
    except RuntimeError:
        model_with_weights = load_model_artifact(checkpoint, MWNet, 'udnn', version='best', user='armlab',
                                                 train_dataset=None, with_joint_positions=with_joint_positions)
        model = model_with_weights.udnn
    model.eval()
    return model


def prepare_data_loaders(batch_size, dataset_dir, take, skip, transform, repeat, no_validate, model_params):
    train_dataset = TorchDynamicsDataset(dataset_dir, mode=model_params['train_mode'], transform=transform)
    train_dataset_take = take_subset(train_dataset, take)
    train_dataset_skip = dataset_skip(train_dataset_take, skip)
    train_dataset_repeat = repeat_dataset(train_dataset_skip, repeat)
    train_dataset_len = len(train_dataset_repeat)
    train_loader = DataLoader(train_dataset_repeat,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=my_collate,
                              num_workers=get_num_workers(batch_size))

    model_params['take'] = take
    model_params['no_validate'] = no_validate
    model_params['dataset_dir'] = dataset_dir
    model_params['train_dataset_size'] = train_dataset_len
    model_params['scenario'] = train_dataset.params['scenario']
    model_params['dataset_hparams'] = train_dataset.params

    val_dataset = TorchDynamicsDataset(dataset_dir, mode=model_params['val_mode'], transform=transform)
    val_dataset_len = len(val_dataset)
    model_params['val_dataset_size'] = val_dataset_len

    val_loader = None
    if val_dataset_len and not no_validate:
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                collate_fn=my_collate,
                                num_workers=get_num_workers(batch_size))

    return train_loader, val_loader


def fine_tune_main(dataset_dir: pathlib.Path,
                   checkpoint: str,
                   model_params: pathlib.Path,
                   batch_size: int,
                   epochs: int,
                   seed: int,
                   user: str,
                   steps: int = -1,
                   nickname: Optional[str] = None,
                   take: Optional[int] = None,
                   skip: Optional[int] = None,
                   repeat: Optional[int] = None,
                   no_validate: bool = False,
                   project=PROJECT,
                   **kwargs):
    pl.seed_everything(seed, workers=True)
    if steps != -1:
        steps = int(steps / batch_size)

    transform = transforms.Compose([])

    model_params = load_hjson(model_params)

    train_loader, val_loader = prepare_data_loaders(batch_size, dataset_dir, take, skip, transform, repeat, no_validate,
                                                    model_params)

    run_id = generate_id(length=5)
    if nickname is not None:
        run_id = nickname + '-' + run_id
    wandb_kargs = {
        'entity': user,
        'resume': True,
    }

    model = load_model_artifact(checkpoint, UDNN, project=project, version='latest', user=user, **model_params)

    wb_logger = WandbLogger(project=project, name=run_id, id=run_id, log_model='all', **wandb_kargs)
    ckpt_cb = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, save_last=True, filename='{epoch:02d}')
    hearbeat_callback = HeartbeatCallback(model.scenario)
    trainer = pl.Trainer(gpus=1,
                         logger=wb_logger,
                         enable_model_summary=False,
                         max_epochs=epochs,
                         max_steps=steps,
                         log_every_n_steps=1,
                         check_val_every_n_epoch=1,
                         callbacks=[ckpt_cb, hearbeat_callback],
                         default_root_dir='wandb',
                         gradient_clip_val=0.05)
    wb_logger.watch(model)
    trainer.fit(model,
                train_loader,
                val_dataloaders=val_loader)
    wandb.finish()
    eval_main(dataset_dir,
              run_id,
              mode='test',
              user=user,
              batch_size=batch_size)
    return run_id


def train_main(dataset_dir: pathlib.Path,
               model_params: pathlib.Path,
               batch_size: int,
               epochs: int,
               seed: int,
               user: str,
               steps: int = -1,
               nickname: Optional[str] = None,
               checkpoint: Optional = None,
               take: Optional[int] = None,
               skip: Optional[int] = None,
               repeat: Optional[int] = None,
               no_validate: bool = False,
               project=PROJECT,
               **kwargs):
    pl.seed_everything(seed, workers=True)
    if steps != -1:
        steps = int(steps / batch_size)

    transform = transforms.Compose([remove_keys("scene_msg")])

    model_params = load_hjson(model_params)

    train_loader, val_loader = prepare_data_loaders(batch_size, dataset_dir, take, skip, transform, repeat, no_validate,
                                                    model_params)

    # add some extra useful info here
    stamp = "{:%B_%d_%H-%M-%S}".format(datetime.now())
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:10]
    model_params['sha'] = sha
    model_params['start-train-time'] = stamp
    model_params['batch_size'] = batch_size
    model_params['seed'] = seed
    model_params['max_epochs'] = epochs
    model_params['max_steps'] = steps
    model_params['checkpoint'] = checkpoint

    if checkpoint is None:
        ckpt_path = None
        run_id = generate_id(length=5)
        if nickname is not None:
            run_id = nickname + '-' + run_id
        wandb_kargs = {'entity': user}
    else:
        ckpt_path = model_artifact_path(checkpoint, project, version='latest', user=user)
        run_id = checkpoint
        wandb_kargs = {
            'entity': user,
            'resume': True,
        }

    model = UDNN(**model_params)
    wb_logger = WandbLogger(project=project, name=run_id, id=run_id, log_model='all', **wandb_kargs)
    ckpt_cb = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, save_last=True, filename='{epoch:02d}')
    hearbeat_callback = HeartbeatCallback(model.scenario)
    trainer = pl.Trainer(gpus=1,
                         logger=wb_logger,
                         enable_model_summary=False,
                         max_epochs=epochs,
                         max_steps=steps,
                         log_every_n_steps=1,
                         check_val_every_n_epoch=1,
                         callbacks=[ckpt_cb, hearbeat_callback],
                         default_root_dir='wandb',
                         gradient_clip_val=0.05)
    wb_logger.watch(model)
    trainer.fit(model,
                train_loader,
                val_dataloaders=val_loader,
                ckpt_path=ckpt_path)
    wandb.finish()
    eval_main(dataset_dir,
              run_id,
              mode='test',
              user=user,
              batch_size=batch_size)
    return run_id


def eval_main(dataset_dir: pathlib.Path,
              checkpoint: str,
              mode: str,
              batch_size: int,
              user: str,
              take: Optional[int] = None,
              skip: Optional[int] = None,
              project=PROJECT,
              **kwargs):
    model = load_udnn_model_wrapper(checkpoint)
    model.eval()
    model.testing = True

    run_id = f'eval-{generate_id(length=5)}'
    eval_config = {
        'training_dataset':       model.hparams.dataset_dir,
        'eval_dataset':           dataset_dir.as_posix(),
        'eval_dataset_versioned': get_dataset_with_version(dataset_dir, PROJECT),
        'eval_checkpoint':        checkpoint,
        'eval_mode':              mode,
    }

    wb_logger = WandbLogger(project=project, name=run_id, id=run_id, tags=['eval'], config=eval_config, entity='armlab')
    trainer = pl.Trainer(gpus=1, enable_model_summary=False, logger=wb_logger)

    transform = transforms.Compose([remove_keys("scene_msg")])
    dataset = TorchDynamicsDataset(dataset_dir, mode, transform=transform)
    dataset = take_subset(dataset, take)
    dataset = dataset_skip(dataset, skip)
    loader = DataLoader(dataset, collate_fn=my_collate, num_workers=get_num_workers(batch_size))
    metrics = trainer.validate(model, loader, verbose=False)
    wandb.finish()

    print(f'run_id: {run_id}')
    for metrics_i in metrics:
        for k, v in metrics_i.items():
            print(f"{k:20s}: {v:0.6f}")

    return metrics


def viz_main(dataset_dir: pathlib.Path,
             checkpoint,
             mode: str,
             skip: Optional[int] = None,
             **kwargs):
    original_dataset = TorchDynamicsDataset(dataset_dir, mode)

    dataset = dataset_skip(original_dataset, skip)

    model = load_udnn_model_wrapper(checkpoint)

    s = original_dataset.get_scenario()

    dataset_anim = RvizAnimationController(n_time_steps=len(dataset), ns='trajs')
    time_anim = RvizAnimationController(n_time_steps=10)

    n_examples_visualized = 0
    while not dataset_anim.done:
        inputs = dataset[dataset_anim.t()]

        if 'meta_mask' in inputs:
            if inputs['meta_mask'].sum() < 6:
                dataset_anim.step()
                continue
            print(inputs['meta_mask'])
        weight = inputs.get('weight', np.ones_like(inputs['time_idx']))
        outputs = remove_batch(model(torchify(add_batch(inputs))))

        time_anim.reset()
        while not time_anim.done:
            t = time_anim.t()
            init_viz_env(s, inputs, t)
            viz_pred_actual_t(original_dataset, model, inputs, outputs, s, t, threshold=0.08)
            s.plot_weight_rviz(weight[t])
            time_anim.step()

        n_examples_visualized += 1

        dataset_anim.step()

    print(f"{n_examples_visualized:=}")
