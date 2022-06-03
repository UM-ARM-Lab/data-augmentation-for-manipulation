#!/usr/bin/env python
import argparse
import pathlib
import tempfile

import hjson
from tqdm import tqdm

from arc_utilities import ros_init
from moonshine.filepath_tools import load_hjson
from moonshine.magic import wandb_lightning_magic
from state_space_dynamics import train_test_dynamics


@ros_init.with_ros("search_threshold")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)

    args = parser.parse_args()

    wandb_lightning_magic()

    thresholds = [
        0.05,
        0.08,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.5,
    ]

    for threshold in tqdm(thresholds):
        test_threshold(args.dataset_dir, threshold, batch_size=64, repeat=5)


def update_params(params_filename, update_dict):
    params = load_hjson(params_filename)
    params.update(update_dict)
    tmp_params_file = tempfile.NamedTemporaryFile('w+', delete=False)
    hjson.dump(params, tmp_params_file)
    return pathlib.Path(tmp_params_file.name)


def test_threshold(dataset_dir, threshold: float, batch_size: int, repeat: int):
    tmp_iterative_lowest_error_params_filename = update_params(pathlib.Path("hparams/iterative_lowest_error.hjson"),
                                                               {'mask_threshold': threshold})
    test_threshold_w_params(dataset_dir, tmp_iterative_lowest_error_params_filename,
                            f'search_iterative_lowset_error_{threshold}', batch_size, repeat)

    tmp_low_initial_error_params_filename = update_params(pathlib.Path("hparams/low_initial_error.hjson"),
                                                          {'mask_threshold': threshold})
    test_threshold_w_params(dataset_dir, tmp_low_initial_error_params_filename,
                            f'search_low_initial_error_{threshold}', batch_size, repeat)


def test_threshold_w_params(dataset_dir: pathlib.Path,
                            params_filename: pathlib.Path,
                            nickname: str,
                            batch_size: int,
                            repeat: int):
    unadapted_checkpoint = 'unadapted-b8s5s'

    fine_tuned_checkpoint = train_test_dynamics.fine_tune_main(dataset_dir=dataset_dir,
                                                               checkpoint=unadapted_checkpoint,
                                                               params_filename=params_filename,
                                                               batch_size=batch_size,
                                                               epochs=-1,
                                                               nickname=nickname,
                                                               steps=500_000,
                                                               user='armlab',
                                                               seed=0,
                                                               repeat=repeat)
    test_dataset_dir = pathlib.Path("known_good_4")
    train_test_dynamics.eval_main(dataset_dir=test_dataset_dir,
                                  checkpoint=fine_tuned_checkpoint,
                                  mode='test',
                                  batch_size=batch_size,
                                  user='armlab')


if __name__ == '__main__':
    main()
