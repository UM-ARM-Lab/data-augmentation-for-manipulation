#!/usr/bin/env python

import pathlib
from datetime import datetime
from typing import Optional

import git
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from wandb import CommError
from wandb.util import generate_id

from link_bot_data.visualization import init_viz_env, viz_pred_actual_t
from link_bot_pycommon.load_wandb_model import load_model_artifact, model_artifact_path
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.filepath_tools import load_hjson
from moonshine.moonshine_utils import get_num_workers
from moonshine.torch_and_tf_utils import add_batch, remove_batch
from moonshine.torch_datasets_utils import take_subset, dataset_skip, my_collate, repeat_dataset
from moonshine.torchify import torchify
from state_space_dynamics.mw_net import MWNet
from state_space_dynamics.mw_net import UDNN
from state_space_dynamics.torch_dynamics_dataset import TorchMetaDynamicsDataset, remove_keys

PROJECT = 'udnn'


def get_dataset_with_version(dataset_dir: pathlib.Path, project=PROJECT, entity='armlab'):
    api = wandb.Api({'entity': entity})
    try:
        artifact = api.artifact(f"{project}/{dataset_dir.name}:latest")
        return artifact.version
    except CommError:
        return 'null'


def train_model_params(batch_size, checkpoint, epochs, model_params_path, seed, steps, take, train_dataset,
                       train_dataset_len):
    model_params = load_hjson(model_params_path)
    model_params['scenario'] = train_dataset.params['scenario']
    model_params['dataset_dir'] = train_dataset.dataset_dir
    model_params['dataset_dir_version'] = get_dataset_with_version(train_dataset.dataset_dir)
    model_params['dataset_hparams'] = train_dataset.params
    # add some extra useful info here
    stamp = "{:%B_%d_%H-%M-%S}".format(datetime.now())
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:10]
    model_params['sha'] = sha
    model_params['start-train-time'] = stamp
    model_params['train_dataset_size'] = train_dataset_len
    model_params['batch_size'] = batch_size
    model_params['seed'] = seed
    model_params['max_epochs'] = epochs
    model_params['max_steps'] = steps
    model_params['take'] = take
    model_params['mode'] = 'train'
    model_params['checkpoint'] = checkpoint
    return model_params


def fine_tune_main(dataset_dir: pathlib.Path,
                   model_params_path: pathlib.Path,
                   checkpoint: str,
                   batch_size: int,
                   epochs: int,
                   seed: int,
                   user: str,
                   steps: int = -1,
                   nickname: Optional[str] = None,
                   take: Optional[int] = None,
                   skip: Optional[int] = None,
                   repeat: Optional[int] = None,
                   project=PROJECT,
                   **kwargs):
    pl.seed_everything(seed, workers=True)
    if steps != -1:
        steps = int(steps / batch_size)

    transform = transforms.Compose([remove_keys("scene_msg", "env", "sdf", "sdf_grad")])
    train_dataset = TorchMetaDynamicsDataset(dataset_dir, transform=transform)
    train_dataset_take = take_subset(train_dataset, take)
    train_dataset_skip = dataset_skip(train_dataset_take, skip)
    train_dataset_repeat = repeat_dataset(train_dataset_skip, repeat)
    train_dataset_len = len(train_dataset_repeat)
    train_loader = DataLoader(train_dataset_repeat,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=my_collate,
                              num_workers=get_num_workers(batch_size))

    run_id = generate_id(length=5)
    if nickname is not None:
        run_id = nickname + '-' + run_id
    wandb_kargs = {
        'entity': user,
        'resume': True,
    }

    # load the udnn checkpoint, create the MWNet, then copy the restored udnn model state into the udnn inside mwnet
    try:
        checkpoint_is_udnn = True
        udnn = load_model_artifact(checkpoint, UDNN, project=project, version='best', user=user)
        model_params = udnn.hparams_initial
        model_params.update(train_model_params(batch_size,
                                               checkpoint,
                                               epochs,
                                               model_params_path,
                                               seed,
                                               steps,
                                               take,
                                               train_dataset,
                                               train_dataset_len,
                                               ))
        model = MWNet(train_dataset=train_dataset, **model_params)
        model.udnn.load_state_dict(udnn.state_dict())
    except Exception:
        checkpoint_is_udnn = False
        model = load_model_artifact(checkpoint, MWNet, project=project, version='latest', user=user,
                                    train_dataset=train_dataset)

    wb_logger = WandbLogger(project=project, name=run_id, id=run_id, log_model='all', **wandb_kargs)
    ckpt_cb = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, save_last=True, filename='{epoch:02d}')
    trainer = pl.Trainer(gpus=1,
                         logger=wb_logger,
                         enable_model_summary=False,
                         max_epochs=epochs,
                         max_steps=steps,
                         log_every_n_steps=1,
                         check_val_every_n_epoch=10,
                         callbacks=[ckpt_cb],
                         num_sanity_val_steps=0,
                         default_root_dir='wandb')
    wb_logger.watch(model)

    if checkpoint_is_udnn:
        model.init_data_weights_from_model_error(train_dataset)
        # model.init_data_weights_from_heuristic(train_dataset)

    trainer.fit(model, train_loader, val_dataloaders=train_loader)
    wandb.finish()
    eval_main(dataset_dir, run_id, mode='test', user=user, batch_size=batch_size)
    return run_id


def train_main(dataset_dir: pathlib.Path,
               model_params_path: pathlib.Path,
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
               project=PROJECT,
               **kwargs):
    pl.seed_everything(seed, workers=True)
    if steps != -1:
        steps = int(steps / batch_size)

    transform = transforms.Compose([remove_keys("scene_msg", "env", "sdf", "sdf_grad")])

    train_dataset = TorchMetaDynamicsDataset(dataset_dir, transform=transform)
    train_dataset_take = take_subset(train_dataset, take)
    train_dataset_skip = dataset_skip(train_dataset_take, skip)
    train_dataset_repeat = repeat_dataset(train_dataset_skip, repeat)
    train_dataset_len = len(train_dataset_repeat)
    train_loader = DataLoader(train_dataset_repeat,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=my_collate,
                              num_workers=get_num_workers(batch_size))

    model_params = train_model_params(batch_size,
                                      checkpoint,
                                      epochs,
                                      model_params_path,
                                      seed,
                                      steps,
                                      take,
                                      train_dataset,
                                      train_dataset_len,
                                      )

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

    model = MWNet(**model_params, train_dataset=train_dataset)
    wb_logger = WandbLogger(project=project, name=run_id, id=run_id, log_model='all', **wandb_kargs)
    ckpt_cb = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, save_last=True, filename='{epoch:02d}')
    trainer = pl.Trainer(gpus=1,
                         logger=wb_logger,
                         enable_model_summary=False,
                         max_epochs=epochs,
                         max_steps=steps,
                         log_every_n_steps=1,
                         check_val_every_n_epoch=4,
                         callbacks=[ckpt_cb],
                         default_root_dir='wandb')
    wb_logger.watch(model)
    trainer.fit(model, train_loader, ckpt_path=ckpt_path)
    wandb.finish()
    eval_main(dataset_dir, run_id, batch_size=batch_size, user=user)
    return run_id


def eval_main(dataset_dir: pathlib.Path,
              checkpoint: str,
              batch_size: int,
              user: str,
              take: Optional[int] = None,
              skip: Optional[int] = None,
              project=PROJECT,
              **kwargs):
    eval_mode = 'test'
    model = load_model_artifact(checkpoint, MWNet, project, version='best', user=user, train_dataset=None)
    model.eval()

    run_id = f'eval-{generate_id(length=5)}'
    eval_config = {
        'training_dataset': model.hparams.dataset_dir,
        'eval_dataset':     dataset_dir.as_posix(),
        'eval_checkpoint':  checkpoint,
        'eval_mode':        eval_mode,
    }

    wb_logger = WandbLogger(project=project, name=run_id, id=run_id, tags=['eval'], config=eval_config, entity='armlab')
    trainer = pl.Trainer(gpus=1, enable_model_summary=False, logger=wb_logger)

    transform = transforms.Compose([remove_keys("scene_msg")])
    dataset = TorchMetaDynamicsDataset(dataset_dir, transform=transform, eval_mode=eval_mode)
    dataset = take_subset(dataset, take)
    dataset = dataset_skip(dataset, skip)
    loader = DataLoader(dataset, collate_fn=my_collate, num_workers=get_num_workers(batch_size))
    metrics = trainer.validate(model, loader, verbose=False)
    wandb.finish()

    print(f'run_id: {run_id}')
    for metrics_i in metrics:
        for k, v in metrics_i.items():
            print(f"{k:20s}: {v:0.5f}")

    return metrics


def viz_main(dataset_dir: pathlib.Path,
             checkpoint,
             user: str,
             weight_above: Optional[float] = None,
             weight_below: Optional[float] = None,
             skip: Optional[int] = None,
             project=PROJECT,
             **kwargs):
    dataset = TorchMetaDynamicsDataset(dataset_dir, eval_mode='test')

    dataset = dataset_skip(dataset, skip)

    model = load_model_artifact(checkpoint, MWNet, project, version='best', user=user, train_dataset=None)
    model.eval()

    s = dataset.get_scenario()

    dataset_anim = RvizAnimationController(n_time_steps=len(dataset), ns='trajs')
    print("Visualizing meta train data")

    n_examples_visualized = 0
    while not dataset_anim.done:
        inputs = dataset[dataset_anim.t()]
        inputs = inputs['meta_train']

        outputs = remove_batch(model(torchify(add_batch(inputs))))
        weight = model.sample_weights.detach().cpu()[inputs['example_idx']]

        if weight_below is not None and bool(weight > weight_below):
            dataset_anim.step()
            continue
        if weight_above is not None and bool(weight < weight_above):
            dataset_anim.step()
            continue

        n_time_steps = inputs['time_idx'].shape[0]
        time_anim = RvizAnimationController(n_time_steps=n_time_steps)

        while not time_anim.done:
            t = time_anim.t()
            init_viz_env(s, inputs, t)
            s.plot_weight_rviz(weight)
            viz_pred_actual_t(dataset, model, inputs, outputs, s, t, threshold=0.05)
            time_anim.step()

        n_examples_visualized += 1

        dataset_anim.step()

    print(f"{n_examples_visualized:=}")
