import argparse
import logging
import pathlib
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

from arc_utilities import ros_init
from link_bot_classifiers.augment_classifier_dataset import augment_classifier_dataset
from link_bot_classifiers.train_test_classifier import train_main, ClassifierEvaluation
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.job_chunking import JobChunker
from moonshine.filepath_tools import load_hjson


@ros_init.with_ros("deep_dive")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-models', '-m', type=int, default=3)
    parser.add_argument('--max-datasets', '-d', type=int, default=3)

    args = parser.parse_args()

    tf.get_logger().setLevel(logging.FATAL)

    aug_hparams_filename = pathlib.Path("hparams/classifier/aug.hjson")
    aug_hparams = load_hjson(aug_hparams_filename)
    no_aug_hparams_filename = pathlib.Path("hparams/classifier/no_aug.hjson")
    scenario = get_scenario("dual_arm_rope_sim_val_with_robot_feasibility_checking")

    base_dataset_dir = pathlib.Path("classifier_data/deep_drive_iters0-2")
    mistake_dataset_dir = pathlib.Path("classifier_data/mistake")

    chunker = JobChunker(logfile_name=pathlib.Path('results/how_many_augmentations.hjson'))

    for n_idx in range(1, 25):
        train_eval_for_k(chunker,
                         n_idx,
                         args.max_models,
                         args.max_datasets,
                         aug_hparams,
                         base_dataset_dir,
                         mistake_dataset_dir,
                         no_aug_hparams_filename,
                         scenario)


def train_eval_for_k(chunker,
                     n_idx: int,
                     max_models: int,
                     max_datasets: int,
                     aug_hparams: Dict,
                     base_dataset_dir: pathlib.Path,
                     mistake_dataset_dir: pathlib.Path,
                     no_aug_hparams_filename: pathlib.Path,
                     scenario):
    for k_idx in range(max_datasets):
        suffix = f"aug-{n_idx}-{k_idx}"
        aug_dataset_dir = base_dataset_dir.parent / f"{base_dataset_dir.name}+{suffix}"

        if not aug_dataset_dir.exists():
            augment_classifier_dataset(dataset_dir=base_dataset_dir,
                                       hparams=aug_hparams,
                                       outdir=aug_dataset_dir,
                                       n_augmentations=n_idx,
                                       scenario=scenario)

        for m_idx in range(max_models):
            train_and_eval_on_mistake(chunker,
                                      n_idx,
                                      m_idx,
                                      k_idx,
                                      aug_dataset_dir,
                                      base_dataset_dir,
                                      mistake_dataset_dir,
                                      no_aug_hparams_filename,
                                      suffix)


def make_key(n: int, m: int, k: int):
    return f"{n}-{m}-{k}"


def train_and_eval_on_mistake(chunker: JobChunker,
                              n_idx: int,
                              m_idx: int,
                              k_idx: int,
                              aug_dataset_dir: pathlib.Path,
                              base_dataset_dir: pathlib.Path,
                              mistake_dataset_dir: pathlib.Path,
                              no_aug_hparams_filename: pathlib.Path,
                              suffix: str):
    key = make_key(n_idx, m_idx, k_idx)
    output = chunker.get_result(key)
    if output is None:
        trial_path, _ = train_main(dataset_dirs=[aug_dataset_dir],
                                   model_hparams=no_aug_hparams_filename,
                                   log=f'{base_dataset_dir.name}-{suffix}',
                                   batch_size=32,
                                   seed=m_idx,
                                   epochs=5)
        view = ClassifierEvaluation(dataset_dirs=[mistake_dataset_dir],
                                    checkpoint=trial_path / 'latest_checkpoint',
                                    mode='all',
                                    batch_size=1,
                                    start_at=0)
        view_it = iter(view)
        batch_idx, example, predictions = next(view_it)
        output = float(predictions['probabilities'][0, 0, 0].numpy())
        chunker.store_result(key, output)


def plot():
    plt.style.use("slides")
    sns.set(rc={'figure.figsize': (7, 4)})

    df = pd.read_csv("results/how_many_augmentations.csv")
    df = df.dropna()
    plt.figure()
    ax = plt.gca()
    sns.lineplot(ax=ax, data=df, x='n_augmentations', y='output', err_style='bars', ci=100)
    sns.scatterplot(ax=ax, data=df, x='n_augmentations', y='output')
    ax.axhline(0.5, color='r')
    ax.set_ylim(-0.01, 1)

    plt.savefig("results/how_many_augmentations.png")
    plt.show()


if __name__ == '__main__':
    main()
    # plot()
