#!/usr/bin/env python
import argparse
import pathlib
from shutil import copy

import numpy as np
from colorama import Fore
from tqdm import tqdm

from link_bot_data.split_dataset import write_mode
from link_bot_data.tf_dataset_utils import pkl_write_example
from link_bot_pycommon.load_wandb_model import load_model_artifact
from moonshine.numpify import numpify
from moonshine.torch_and_tf_utils import add_batch, remove_batch
from moonshine.torchify import torchify
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset
from state_space_dynamics.udnn_torch import UDNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('checkpoint')
    parser.add_argument('threshold', type=float)

    args = parser.parse_args()

    outdir: pathlib.Path = args.dataset_dir.parent / f'{args.dataset_dir.name}+meta_val'
    outdir.mkdir(exist_ok=True, parents=False)

    copy(args.dataset_dir / 'hparams.hjson', outdir / 'hparams.hjson')

    def _copy_dataset_mode_if(mode, predicate):
        dataset = TorchDynamicsDataset(args.dataset_dir, mode=mode)
        print(Fore.CYAN + mode + Fore.RESET)
        files_written = []
        for e in tqdm(dataset):
            example_idx = e['example_idx']
            if predicate(e):
                _, full_filename = pkl_write_example(outdir, e, example_idx)
                files_written.append(full_filename)
            else:
                print(f"dropping {example_idx}")
        return files_written

    model = load_model_artifact(args.checkpoint, UDNN, project='udnn', version='latest', user='armlab')

    # then for the test and validation sets...
    # if the error is below the threshold, add it to the validation set
    def _low_model_error(actual):
        predictions = numpify(remove_batch(model(torchify(add_batch(actual)))))
        error = model.scenario.classifier_distance(actual, predictions)
        return np.all(error < args.threshold)

    val_files = _copy_dataset_mode_if('val', _low_model_error)
    write_mode(outdir, val_files, 'val')
    test_files = _copy_dataset_mode_if('test', _low_model_error)
    write_mode(outdir, test_files, 'test')

    # copy the training set exactly
    train_files = _copy_dataset_mode_if('train', lambda e: True)
    write_mode(outdir, train_files, 'train')



if __name__ == '__main__':
    main()
