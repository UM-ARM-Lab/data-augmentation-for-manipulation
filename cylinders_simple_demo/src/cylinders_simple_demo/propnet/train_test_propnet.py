#!/usr/bin/env python

import pathlib
from typing import Optional

import pytorch_lightning as pl
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms

from cylinders_simple_demo.propnet.propnet_models import PropNet
from cylinders_simple_demo.utils.data_utils import remove_keys, get_num_workers, my_collate, remove_batch, add_batch
from cylinders_simple_demo.utils.my_torch_dataset import MyTorchDataset
from cylinders_simple_demo.utils.torch_datasets_utils import dataset_skip, dataset_repeat
from cylinders_simple_demo.utils.torchify import torchify
from cylinders_simple_demo.utils.utils import load_hjson


def train_main(dataset_dir: pathlib.Path,
               model_params: pathlib.Path,
               nickname: str,
               batch_size: int,
               epochs: int,
               seed: int,
               ):
    pl.seed_everything(seed, workers=True)

    transform = transforms.Compose([
        remove_keys('filename', 'full_filename', 'joint_names', 'metadata', 'is_valid', 'augmented_from'),
    ])

    train_dataset = MyTorchDataset(dataset_dir, mode='train', transform=transform)
    val_dataset = MyTorchDataset(dataset_dir, mode='val', transform=transform)

    train_dataset_repeated = dataset_repeat(train_dataset, 10)
    train_loader = DataLoader(train_dataset_repeated,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=my_collate,
                              num_workers=get_num_workers(batch_size))

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            collate_fn=my_collate,
                            num_workers=get_num_workers(batch_size))

    model_params = load_hjson(model_params)
    model_params['num_objects'] = train_dataset.params['data_collection_params']['num_objs'] + 1
    model_params['scenario'] = train_dataset.params['scenario']
    # add some extra useful info here
    model_params['n_train_trajs'] = train_dataset.params['n_train_trajs']
    model_params['used_augmentation'] = train_dataset.params.get('used_augmentation', False)
    model_params['n_augmentations'] = train_dataset.params.get('n_augmentations', None)
    model_params['train_dataset_size'] = len(train_dataset)
    model_params['val_dataset_size'] = len(val_dataset)
    model_params['batch_size'] = batch_size
    model_params['seed'] = seed
    model_params['max_epochs'] = epochs
    model_params['mode'] = 'train'

    model = PropNet(hparams=model_params)

    ckpt_cb = ModelCheckpoint(monitor="val_loss", save_top_k=1, save_last=True, filename='{epoch:02d}')
    tb_logger = TensorBoardLogger(save_dir='tb_logs', name=nickname)

    trainer = pl.Trainer(gpus=1,
                         enable_model_summary=False,
                         logger=tb_logger,
                         max_epochs=epochs,
                         log_every_n_steps=10,
                         check_val_every_n_epoch=1,
                         callbacks=[ckpt_cb],
                         gradient_clip_val=0.1)

    trainer.fit(model, train_loader, val_dataloaders=val_loader)


def eval_main(dataset_dir: pathlib.Path, checkpoint: str, mode: str, batch_size: int):
    dataset = MyTorchDataset(dataset_dir, mode)
    loader = DataLoader(dataset, collate_fn=my_collate, num_workers=get_num_workers(batch_size), batch_size=batch_size)

    model = PropNet.load_from_checkpoint(checkpoint_path=checkpoint)

    trainer = pl.Trainer(gpus=1, enable_model_summary=False)
    trainer.validate(model=model, dataloaders=loader)


def viz_main(dataset_dir: pathlib.Path, checkpoint, mode: str, skip: Optional[int] = None):
    dataset = MyTorchDataset(dataset_dir, mode)
    s = dataset.get_scenario()

    dataset_ = dataset_skip(dataset, skip)

    model = PropNet.load_from_checkpoint(checkpoint_path=checkpoint)
    model.eval()

    for example in dataset_:
        model_inputs = torchify(add_batch(example))
        gt_vel, gt_pos, pred_vel, pred_pos = remove_batch(*model(model_inputs))
        anim = s.example_and_predictions_to_animation(example, gt_vel, gt_pos, pred_vel, pred_pos)
        plt.show()
