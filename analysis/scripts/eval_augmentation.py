#!/usr/bin/env python
import argparse
import logging
import pathlib

import tensorflow as tf
from tqdm import tqdm

from arc_utilities import ros_init
from augmentation.add_augmentation_configs import add_augmentation_configs_to_dataset
from link_bot_classifiers.get_model import get_model
from link_bot_data.load_dataset import get_classifier_dataset_loader
from moonshine import common_train_hparams, filepath_tools
from moonshine.filepath_tools import load_hjson


def eval_augmentation(dataset_dir: pathlib.Path,
                      hparams: pathlib.Path,
                      nickname: str,
                      new_env_dir: pathlib.Path,
                      n_augmentations: int,
                      seed: int):
    batch_size = 8
    trials_directory = pathlib.Path("./cl_trials").absolute()
    dataset_dirs = [dataset_dir]
    model_hparams = load_hjson(hparams)
    model_class = get_model(model_hparams['model_class'])

    train_dataset_loader = get_classifier_dataset_loader(dataset_dirs=dataset_dirs)

    model_hparams['classifier_dataset_hparams'] = train_dataset_loader.hparams
    model_hparams.update(common_train_hparams.setup_hparams(batch_size, dataset_dirs, seed, train_dataset_loader))
    model = model_class(hparams=model_hparams, batch_size=batch_size, scenario=train_dataset_loader.get_scenario())

    trial_path, _ = filepath_tools.create_or_load_trial(group_name=pathlib.Path(nickname),
                                                        params=model_hparams,
                                                        trials_directory=trials_directory)

    model.save_inputs_path = trial_path / 'saved_inputs'

    train_dataset = train_dataset_loader.get_datasets(mode='all', shuffle=False)
    train_dataset = train_dataset.batch(batch_size)

    train_dataset = add_augmentation_configs_to_dataset(new_env_dir, train_dataset, batch_size)

    for i in range(n_augmentations):
        for e in tqdm(train_dataset):
            e['batch_size'] = batch_size
            out = model.preprocess_no_gradient(e, training=True)
            out['is_valid']
            print(out.keys())


@ros_init.with_ros("eval_augmentation")
def main():
    tf.get_logger().setLevel(logging.FATAL)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('hparams', type=pathlib.Path)
    parser.add_argument('nickname')
    parser.add_argument('--new-env-dir', type=pathlib.Path,
                        default=pathlib.Path('/media/shared/pretransfer_initial_configs/car3/'))
    parser.add_argument('--n-augmentations', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    eval_augmentation(dataset_dir=args.dataset_dir,
                      hparams=args.hparams,
                      nickname=args.nickname,
                      new_env_dir=args.new_env_dir,
                      n_augmentations=args.n_augmentations,
                      seed=args.seed)


if __name__ == '__main__':
    main()
