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
from tqdm import tqdm
from wandb.util import generate_id

from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.filepath_tools import load_hjson
from moonshine.numpify import numpify
from moonshine.torch_utils import my_collate
from propnet.propnet_models import PropNet
from propnet.torch_dynamics_dataset import TorchDynamicsDataset, remove_keys

PROJECT = 'propnet'


class MyModelCheckpoint(pl.callbacks.ModelCheckpoint):

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        return {
            "monitor":             self.monitor,
            "best_model_score":    self.best_model_score,
            "best_model_path":     self.best_model_path,
            "current_score":       self.current_score,
            "dirpath":             self.dirpath,
            "best_k_models":       self.best_k_models,
            "kth_best_model_path": self.kth_best_model_path,
        }

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        self.best_model_score = callback_state["best_model_score"]
        self.best_model_path = callback_state["best_model_path"]
        self.best_model_score = callback_state["best_model_score"]
        self.best_k_models = callback_state["best_k_models"]
        self.kth_best_model_path = callback_state["kth_best_model_path"]


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
               no_validate: bool = False,
               project=PROJECT,
               **kwargs):
    pl.seed_everything(seed, workers=True)

    transform = transforms.Compose([
        remove_keys('filename', 'full_filename', 'joint_names', 'metadata', 'is_valid', 'augmented_from'),
    ])

    train_dataset = TorchDynamicsDataset(dataset_dir, mode='train',
                                         transform=transform)
    val_dataset = TorchDynamicsDataset(dataset_dir, mode='val',
                                       transform=transform)

    train_dataset_take = take_subset(train_dataset, take)

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
    model_params['n_train_trajs'] = train_dataset.params['n_train_trajs']
    model_params['used_augmentation'] = train_dataset.params.get('used_augmentation', False)
    model_params['n_augmentations'] = train_dataset.params.get('n_augmentations', None)
    model_params['train_dataset_size'] = len(train_dataset_take)
    model_params['val_dataset_size'] = len(val_dataset)
    model_params['batch_size'] = batch_size
    model_params['seed'] = seed
    model_params['max_epochs'] = epochs
    model_params['max_steps'] = steps
    model_params['take'] = take
    model_params['mode'] = 'train'
    model_params['checkpoint'] = checkpoint
    model_params['no_validate'] = no_validate

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

    model = PropNet(hparams=model_params)

    wb_logger = WandbLogger(project=project, name=run_id, id=run_id, log_model='all', **wandb_kargs)

    ckpt_cb = MyModelCheckpoint(monitor="val_loss", save_top_k=1, save_last=True, filename='{epoch:02d}')

    trainer = pl.Trainer(gpus=1,
                         logger=wb_logger,
                         enable_model_summary=False,
                         max_epochs=epochs,
                         max_steps=steps,
                         log_every_n_steps=1,
                         check_val_every_n_epoch=10,
                         callbacks=[ckpt_cb],
                         default_root_dir='wandb',
                         gradient_clip_val=0.1)

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


def take_subset(dataset, take):
    if take is None:
        return dataset

    dataset_take = Subset(dataset, range(min(take, len(dataset))))
    return dataset_take


def eval_main(dataset_dir: pathlib.Path,
              checkpoint: str,
              mode: str,
              batch_size: int,
              user: str,
              take: int = None,
              project=PROJECT,
              **kwargs):
    model = load_model_artifact(checkpoint, PropNet, project, version='best', user=user)

    run_id = f'eval-{generate_id(length=5)}'
    eval_config = {
        'training_dataset': model.hparams.dataset_dir,
        'eval_dataset':     dataset_dir.as_posix(),
        'eval_checkpoint':  checkpoint,
        'eval_mode':        mode,
    }

    wb_logger = WandbLogger(project=project, name=run_id, id=run_id, tags=['eval'], config=eval_config, entity=user)
    trainer = pl.Trainer(gpus=1, enable_model_summary=False, logger=wb_logger)

    dataset = TorchDynamicsDataset(dataset_dir, mode)
    dataset = take_subset(dataset, take)
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
             mode: str,
             user: str,
             project=PROJECT,
             **kwargs):
    dataset = TorchDynamicsDataset(dataset_dir, mode)
    s = dataset.get_scenario()

    loader = DataLoader(dataset, collate_fn=my_collate)

    model = load_model_artifact(checkpoint, PropNet, project, version='best', user=user)
    model.training = False

    for i, inputs in enumerate(tqdm(loader)):
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

            pred_state_t = s.propnet_outputs_to_state(inputs=inputs, pred_vel=pred_vel, pred_pos=pred_pos, b=b, t=t,
                                                      obj_dz=0.01)

            s.plot_state_rviz(pred_state_t, label='predicted', color='#0000ffaa')

            anim.step()


def get_num_workers(batch_size):
    return min(batch_size, multiprocessing.cpu_count())


def load_model_artifact(checkpoint, model_class, project, version, user='armlab'):
    local_ckpt_path = model_artifact_path(checkpoint, project, version, user)
    model = model_class.load_from_checkpoint(local_ckpt_path.as_posix())
    return model


def model_artifact_path(checkpoint, project, version, user='armlab'):
    if ':' in checkpoint:
        checkpoint, version = checkpoint.split(':')

    if not checkpoint.startswith('model-'):
        checkpoint = 'model-' + checkpoint
    api = wandb.Api()
    artifact = api.artifact(f'{user}/{project}/{checkpoint}:{version}')
    artifact_dir = artifact.download()
    local_ckpt_path = pathlib.Path(artifact_dir) / "model.ckpt"
    print(f"Found artifact {local_ckpt_path}")
    return local_ckpt_path
