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
from wandb.util import generate_id

from link_bot_data.visualization import init_viz_env
from link_bot_pycommon.load_wandb_model import load_model_artifact
from mde.mde_weighted_dynamics import MDEWeightedDynamics
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.filepath_tools import load_hjson
from moonshine.moonshine_utils import get_num_workers
from moonshine.my_pl_callbacks import HeartbeatCallback
from moonshine.torch_and_tf_utils import add_batch, remove_batch
from moonshine.torch_datasets_utils import take_subset, dataset_skip, my_collate
from moonshine.torchify import torchify
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset
from state_space_dynamics.torch_dynamics_dataset import remove_keys
from state_space_dynamics.train_test_dynamics import prepare_train, prepare_validation
from state_space_dynamics.udnn_torch import UDNN

PROJECT = 'joint'


def train_main(dataset_dir: pathlib.Path,
               model_params: pathlib.Path,
               checkpoint: str,
               batch_size: int,
               epochs: int,
               seed: int,
               user: str,
               steps: int = -1,
               nickname: Optional[str] = None,
               take: Optional[int] = None,
               skip: Optional[int] = None,
               no_validate: bool = False,
               project=PROJECT,
               **kwargs):
    pl.seed_everything(seed, workers=True)

    transform = transforms.Compose([remove_keys("scene_msg")])

    train_loader, train_dataset, train_dataset_len = prepare_train(batch_size=batch_size, dataset_dir=dataset_dir,
                                                                   take=take, skip=skip, transform=transform,
                                                                   repeat=None)
    val_dataset_len, val_loader = prepare_validation(batch_size=batch_size, dataset_dir=dataset_dir,
                                                     no_validate=no_validate, transform=transform)

    model_params = load_hjson(model_params)
    model_params['scenario'] = train_dataset.params['scenario']
    model_params['dataset_dir'] = train_dataset.dataset_dir
    model_params['dataset_hparams'] = train_dataset.params
    # add some extra useful info here
    stamp = "{:%B_%d_%H-%M-%S}".format(datetime.now())
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:10]
    model_params['sha'] = sha
    model_params['start-train-time'] = stamp
    model_params['train_dataset_size'] = train_dataset_len
    model_params['val_dataset_size'] = val_dataset_len
    model_params['batch_size'] = batch_size
    model_params['seed'] = seed
    model_params['max_epochs'] = epochs
    model_params['max_steps'] = steps
    model_params['take'] = take
    model_params['mode'] = 'train'
    model_params['checkpoint'] = checkpoint
    model_params['no_validate'] = no_validate

    udnn = load_model_artifact(checkpoint, UDNN, project='udnn', version='best', user=user)
    run_id = nickname + '-' + generate_id(length=5)

    model = MDEWeightedDynamics(**model_params)

    # Initializes the mde_weighted_dynamics model with the UDNN
    model.udnn.load_state_dict(udnn.state_dict())

    wandb_kargs = {'entity': user, 'resume': True}
    wb_logger = WandbLogger(project=project, name=run_id, id=run_id, log_model='all', **wandb_kargs)
    ckpt_cb = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, save_last=True, filename='{epoch:02d}')
    hearbeat_callback = HeartbeatCallback(train_dataset.get_scenario())
    trainer = pl.Trainer(gpus=1,
                         logger=wb_logger,
                         enable_model_summary=False,
                         max_epochs=epochs,
                         max_steps=steps,
                         log_every_n_steps=5,
                         check_val_every_n_epoch=1,
                         callbacks=[ckpt_cb, hearbeat_callback],
                         default_root_dir='wandb')
    wb_logger.watch(model)
    trainer.fit(model, train_loader, val_dataloaders=val_loader)
    wandb.finish()

    # script = model.to_torchscript()
    # torch.jit.save(script, "model.pt")

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
    model = load_model_artifact(checkpoint, MDEWeightedDynamics, project, version='best', user=user)

    transform = transforms.Compose([remove_keys("scene_msg")])
    dataset = TorchDynamicsDataset(dataset_dir, mode=mode, transform=transform)
    dataset = take_subset(dataset, take)
    dataset = dataset_skip(dataset, skip)

    run_id = f'eval-{generate_id(length=5)}'
    eval_config = {
        'training_dataset': model.hparams.dataset_dir,
        'eval_dataset':     dataset_dir.as_posix(),
        'eval_checkpoint':  checkpoint,
        'eval_mode':        mode,
    }

    wb_logger = WandbLogger(project=project, name=run_id, id=run_id, tags=['eval'], config=eval_config, entity='armlab')
    trainer = pl.Trainer(gpus=1, enable_model_summary=False, logger=wb_logger)

    loader = DataLoader(dataset, collate_fn=my_collate, num_workers=get_num_workers(batch_size), batch_size=batch_size)
    metrics = trainer.validate(model, loader, verbose=False)
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
    model = load_model_artifact(checkpoint, MDEWeightedDynamics, project, version='best', user=user)
    model.training = False

    dataset = TorchDynamicsDataset(dataset_dir, mode=mode)

    dataset = dataset_skip(dataset, skip)

    s = dataset.get_scenario()

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
