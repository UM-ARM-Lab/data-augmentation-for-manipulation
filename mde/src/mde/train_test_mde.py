#!/usr/bin/env python

import pathlib
from datetime import datetime
from typing import Optional

import git
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from wandb.util import generate_id

from link_bot_data.new_dataset_utils import fetch_mde_dataset
from link_bot_data.visualization import init_viz_env
from link_bot_data.wandb_datasets import get_dataset_with_version
from link_bot_pycommon.load_wandb_model import load_model_artifact, model_artifact_path
from mde.mde_data_module import MDEDataModule
from mde.mde_torch import MDE
from mde.torch_mde_dataset import TorchMDEDataset
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.filepath_tools import load_hjson
from moonshine.my_pl_callbacks import HeartbeatCallback
from moonshine.torch_and_tf_utils import add_batch, remove_batch
from moonshine.torch_datasets_utils import dataset_skip
from moonshine.torchify import torchify

PROJECT = 'mde'


def train_main(dataset_dir: pathlib.Path,
               params_filename: pathlib.Path,
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

    params = load_hjson(params_filename)

    data_module = MDEDataModule(dataset_dir,
                                batch_size=batch_size,
                                take=take,
                                skip=skip,
                                repeat=repeat)
    data_module.add_dataset_params(params)

    # add some extra useful info here
    stamp = "{:%B_%d_%H-%M-%S}".format(datetime.now())
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:10]
    params['sha'] = sha
    params['start-train-time'] = stamp
    params['batch_size'] = batch_size
    params['seed'] = seed
    params['epochs'] = epochs
    params['steps'] = steps
    params['checkpoint'] = checkpoint

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

    model = MDE(**params)
    wb_logger = WandbLogger(project=project, name=run_id, id=run_id, log_model='all', **wandb_kargs)
    ckpt_cb = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, save_last=True, filename='{epoch:02d}')
    hearbeat_callback = HeartbeatCallback(model.scenario)
    max_steps = int(steps / batch_size) if steps != -1 else steps
    print(f"{max_steps=}")
    trainer = pl.Trainer(gpus=1,
                         logger=wb_logger,
                         enable_model_summary=False,
                         max_epochs=epochs,
                         max_steps=max_steps,
                         log_every_n_steps=1,
                         check_val_every_n_epoch=1,
                         callbacks=[ckpt_cb, hearbeat_callback],
                         default_root_dir='wandb')
    wb_logger.watch(model)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)
    wandb.finish()

    # script = model.to_torchscript()
    # torch.jit.save(script, "model.pt")

    # eval_main(dataset_dir,
    #           run_id,
    #           mode='test',
    #           user=user,
    #           batch_size=batch_size)

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
    model = load_model_artifact(checkpoint, MDE, project, version='best', user=user)
    model.eval()

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

    data_module = MDEDataModule(dataset_dir, batch_size=batch_size, take=take, skip=skip)

    metrics = trainer.test(model, data_module, verbose=False)
    wandb.finish()

    print(f'run_id: {run_id}')
    for metrics_i in metrics:
        for k, v in metrics_i.items():
            print(f"{k:20s}: {v:0.5f}")

    return metrics


def viz_main(dataset_dir: pathlib.Path,
             checkpoint,
             mode: str,
             user: str,
             skip: Optional[int] = None,
             project=PROJECT,
             **kwargs):
    model = load_model_artifact(checkpoint, MDE, project, version='best', user=user)
    model.training = False

    dataset = TorchMDEDataset(fetch_mde_dataset(dataset_dir), mode=mode)

    dataset = dataset_skip(dataset, skip)

    s = model.scenario

    dataset_anim = RvizAnimationController(n_time_steps=len(dataset), ns='trajs')
    time_anim = RvizAnimationController(n_time_steps=2)

    n_examples_visualized = 0
    while not dataset_anim.done:
        inputs = dataset[dataset_anim.t()]

        inputs_batch = torchify(add_batch(inputs))
        predicted_error = model(inputs_batch)
        predicted_error = remove_batch(predicted_error)

        time_anim.reset()
        while not time_anim.done:
            t = time_anim.t()
            init_viz_env(s, inputs, t)
            dataset.transition_viz_t()(s, inputs, t)
            s.plot_pred_error_rviz(predicted_error)
            time_anim.step()

            n_examples_visualized += 1

        dataset_anim.step()

    print(f"{n_examples_visualized:=}")
