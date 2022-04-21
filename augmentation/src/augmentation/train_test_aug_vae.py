#!/usr/bin/env python
import pathlib
from datetime import datetime
from typing import Optional, List

import git
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from wandb.util import generate_id

from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.load_wandb_model import model_artifact_path, load_model_artifact
from mde.torch_mde_dataset import TorchMDEDataset
from moonshine.dynamics_aes import DynamicsVAE
from moonshine.filepath_tools import load_hjson
from moonshine.moonshine_utils import get_num_workers
from moonshine.my_torch_dataset import MyTorchDataset
from moonshine.numpify import numpify
from moonshine.torch_and_tf_utils import remove_batch
from moonshine.torch_datasets_utils import take_subset, my_collate, repeat_dataset
from moonshine.torchify import torchify
from moonshine.vae import MyVAE
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset, remove_keys

PROJECT = 'aug_vae'


def fine_tune(dataset_dirs: List[pathlib.Path],
              model_params_path: pathlib.Path,
              epochs: int,
              seed: int,
              batch_size: int = 32,
              user: str = 'armlab',
              steps: int = -1,
              nickname: Optional[str] = None,
              checkpoint: Optional = None,
              no_validate: bool = False,
              scenario: Optional = None,
              project=PROJECT,
              **kwargs):
    pl.seed_everything(seed, workers=True)

    model_params = load_hjson(model_params_path)
    if scenario is None:
        scenario = get_scenario(model_params['scenario'], {'rope_name': 'rope_3d'})

    transform = transforms.Compose([
        remove_keys('filename', 'full_filename', 'joint_names', 'metadata', 'is_valid', 'augmented_from', 'error',
                    'predicted/error', 'sdf', 'sdf_grad'),
    ])

    train_dataset = MyTorchDataset(dataset_dirs, mode='train',
                                   transform=transform)
    train_dataset_repeated = repeat_dataset(train_dataset, repeat=int(max(100 * batch_size / len(train_dataset), 1)))
    val_dataset = MyTorchDataset(dataset_dirs, mode='train',
                                 transform=transform)

    train_loader = DataLoader(train_dataset_repeated,
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

    model_params['scenario'] = train_dataset.params['scenario']
    # add some extra useful info here
    stamp = "{:%B_%d_%H-%M-%S}".format(datetime.now())
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:10]
    model_params['sha'] = sha
    model_params['start-train-time'] = stamp
    model_params['dataset_dir'] = dataset_dirs[-1].as_posix()
    model_params['used_augmentation'] = train_dataset.params.get('used_augmentation', False)
    model_params['n_augmentations'] = train_dataset.params.get('n_augmentations', None)
    model_params['train_dataset_size'] = len(train_dataset)
    model_params['val_dataset_size'] = len(val_dataset)
    model_params['batch_size'] = batch_size
    model_params['seed'] = seed
    model_params['max_epochs'] = epochs
    model_params['max_steps'] = steps
    model_params['mode'] = 'train'
    model_params['checkpoint'] = checkpoint
    model_params['no_validate'] = no_validate

    if checkpoint is None:
        ckpt_path = None
        run_id = nickname
        wandb_kargs = {'entity': user}
    else:
        ckpt_path = model_artifact_path(checkpoint, project, version='latest', user=user)
        run_id = checkpoint
        wandb_kargs = {
            'entity': user,
            'resume': True,
        }

    model = MyVAE(scenario=scenario, hparams=model_params)

    wb_logger = WandbLogger(project=project, name=run_id, id=run_id, log_model='all', **wandb_kargs)

    ckpt_cb = ModelCheckpoint(monitor="val_loss", save_top_k=1, save_last=True, filename='{epoch:02d}')

    trainer = pl.Trainer(gpus=1,
                         logger=wb_logger,
                         enable_model_summary=False,
                         max_epochs=epochs,
                         max_steps=steps,
                         log_every_n_steps=1,
                         check_val_every_n_epoch=1,
                         callbacks=[ckpt_cb],
                         default_root_dir='wandb',
                         gradient_clip_val=0.1)

    wb_logger.watch(model)

    trainer.fit(model,
                train_loader,
                val_dataloaders=val_loader,
                ckpt_path=ckpt_path)
    wandb.finish()

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
               no_validate: bool = False,
               scenario: Optional = None,
               project=PROJECT,
               **kwargs):
    pl.seed_everything(seed, workers=True)

    model_params = load_hjson(model_params_path)
    if scenario is None:
        scenario = get_scenario(model_params['scenario'])

    transform = transforms.Compose([
        remove_keys('filename', 'full_filename', 'joint_names', 'metadata', 'is_valid', 'augmented_from'),
    ])

    train_dataset = MyTorchDataset(dataset_dir, mode='train',
                                   transform=transform)
    val_dataset = MyTorchDataset(dataset_dir, mode='val',
                                 transform=transform)

    train_dataset_take = take_subset(train_dataset, take)

    train_loader = DataLoader(train_dataset_take,
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

    model_params['num_objects'] = train_dataset.params['data_collection_params']['num_objs'] + 1
    model_params['scenario'] = train_dataset.params['scenario']
    # add some extra useful info here
    stamp = "{:%B_%d_%H-%M-%S}".format(datetime.now())
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:10]
    model_params['sha'] = sha
    model_params['start-train-time'] = stamp
    model_params['dataset_dir'] = dataset_dir.as_posix()
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

    model = DynamicsVAE(scenario=scenario, hparams=model_params)

    wb_logger = WandbLogger(project=project, name=run_id, id=run_id, log_model='all', **wandb_kargs)

    ckpt_cb = ModelCheckpoint(monitor="val_loss", save_top_k=1, save_last=True, filename='{epoch:02d}')

    trainer = pl.Trainer(gpus=1,
                         logger=wb_logger,
                         enable_model_summary=False,
                         max_epochs=epochs,
                         max_steps=steps,
                         log_every_n_steps=1,
                         check_val_every_n_epoch=50,
                         callbacks=[ckpt_cb],
                         default_root_dir='wandb',
                         gradient_clip_val=0.1)

    wb_logger.watch(model)

    trainer.fit(model,
                train_loader,
                val_dataloaders=val_loader,
                ckpt_path=ckpt_path)
    wandb.finish()

    return run_id


def eval_main(dataset_dir: pathlib.Path,
              checkpoint: str,
              mode: str,
              batch_size: int,
              user: str,
              take: int = None,
              project=PROJECT,
              **kwargs):
    model = load_model_artifact(checkpoint, DynamicsVAE, project, version='best', user=user)
    model.eval()

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
             mode: str = 'train',
             user: str = 'armlab',
             project=PROJECT,
             **kwargs):
    dataset = TorchMDEDataset(dataset_dir, mode)
    s = dataset.get_scenario({'rope_name': 'rope_3d'})

    loader = DataLoader(dataset, collate_fn=my_collate)

    model = load_model_artifact(checkpoint, MyVAE, project, version='best', user=user)
    model.scenario = s
    model.eval()

    for inputs in tqdm(loader):
        inputs = torchify(inputs)
        outputs = model(inputs)
        inputs_no_batch = numpify(remove_batch(inputs))
        outputs = numpify(remove_batch(outputs))

        dataset.viz_pred_actual(actual=inputs_no_batch, pred=outputs)
