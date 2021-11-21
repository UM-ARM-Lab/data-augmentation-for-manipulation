#!/usr/bin/env python

import multiprocessing
import pathlib
from datetime import datetime
from typing import Optional

import git
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from wandb.util import generate_id

from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.filepath_tools import load_hjson
from moonshine.numpify import numpify
from moonshine.torch_utils import my_collate
from propnet.propnet_models import PropNet
from propnet.torch_dynamics_dataset import TorchDynamicsDataset, remove_keys

PROJECT = 'propnet'


def train_main(dataset_dir: pathlib.Path,
               model_params: pathlib.Path,
               batch_size: int,
               epochs: int,
               seed: int,
               steps: Optional[int] = None,
               checkpoint: Optional = None,
               take: Optional[int] = None,
               no_validate: bool = False,
               project=PROJECT,
               **kwargs):
    pl.seed_everything(seed, workers=True)

    transform = transforms.Compose([
        remove_keys('filename', 'full_filename', 'joint_names', 'metadata'),
    ])

    train_dataset = TorchDynamicsDataset(dataset_dir, mode='train',
                                         transform=transform)
    val_dataset = TorchDynamicsDataset(dataset_dir, mode='val',
                                       transform=transform)

    if take:
        train_dataset_take = take_subset(train_dataset, take)
    else:
        train_dataset_take = train_dataset

    train_loader = DataLoader(train_dataset_take,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=get_num_workers(batch_size))

    val_loader = None
    if len(val_dataset) > 0 and not no_validate:
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=get_num_workers(batch_size))

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
    model_params['train_dataset_size'] = len(train_dataset_take)
    model_params['val_dataset_size'] = len(val_dataset)
    model_params['batch_size'] = batch_size
    model_params['seed'] = seed
    model_params['max_epochs'] = epochs
    model_params['take'] = take
    model_params['checkpoint'] = checkpoint
    model_params['no_validate'] = no_validate

    if checkpoint is None:
        ckpt_path = None
    else:
        ckpt_path = model_artifact_path(checkpoint, project, version='latest', user='petermitrano')

    model = PropNet(hparams=model_params)

    run_id = generate_id(length=5)
    wb_logger = WandbLogger(project=project, name=run_id, id=run_id, log_model='all')
    loggers = [
        wb_logger,
    ]

    ckpt_cb = pl.callbacks.ModelCheckpoint(monitor="val_loss",
                                           save_top_k=-1,
                                           filename='latest-{epoch:02d}',
                                           save_on_train_epoch_end=True)
    early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss",
                                                divergence_threshold=5e-3,
                                                patience=200)
    callbacks = [
        ckpt_cb,
        early_stopping
    ]

    trainer = pl.Trainer(gpus=1,
                         logger=loggers,
                         enable_model_summary=False,
                         max_epochs=epochs,
                         max_steps=steps,
                         log_every_n_steps=1,
                         check_val_every_n_epoch=10,
                         callbacks=callbacks,
                         default_root_dir='wandb',
                         gradient_clip_val=0.1)

    wb_logger.watch(model)

    trainer.fit(model,
                train_loader,
                val_dataloaders=val_loader,
                ckpt_path=ckpt_path)


def take_subset(dataset, take):
    dataset_take = Subset(dataset, range(min(take, len(dataset))))
    return dataset_take


def eval_main(dataset_dir: pathlib.Path,
              checkpoint: pathlib.Path,
              mode: str,
              batch_size: int,
              project=PROJECT,
              **kwargs):
    dataset = TorchDynamicsDataset(dataset_dir, mode)

    loader = DataLoader(dataset, collate_fn=my_collate, num_workers=get_num_workers(batch_size))

    model = load_model_artifact(checkpoint, PropNet, project, 'best')

    trainer = pl.Trainer(gpus=1, enable_model_summary=False)

    metrics = trainer.validate(model, loader, verbose=0)

    for metrics_i in metrics:
        for k, v in metrics_i.items():
            print(f"{k:20s}: {v:0.4f}")

    return metrics


def viz_main(dataset_dir: pathlib.Path,
             checkpoint: pathlib.Path,
             mode: str,
             project=PROJECT,
             **kwargs):
    dataset = TorchDynamicsDataset(dataset_dir, mode)
    s = dataset.get_scenario()

    loader = DataLoader(dataset, collate_fn=my_collate)

    model = load_model_artifact(checkpoint, PropNet, project, 'best')

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


def load_model_artifact(checkpoint, model_class, project, version, user='petermitrano'):
    local_ckpt_path = model_artifact_path(checkpoint, project, version, user)
    model = model_class.load_from_checkpoint(local_ckpt_path.as_posix())
    return model


def model_artifact_path(checkpoint, project, version, user='petermitrano'):
    if not checkpoint.startswith('model-'):
        checkpoint = 'model-' + checkpoint
    api = wandb.Api()
    artifact = api.artifact(f'{user}/{project}/{checkpoint}:{version}')
    artifact_dir = artifact.download()
    local_ckpt_path = pathlib.Path(artifact_dir) / "model.ckpt"
    return local_ckpt_path
