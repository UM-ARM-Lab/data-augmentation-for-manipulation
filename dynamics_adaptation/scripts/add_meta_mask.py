#!/usr/bin/env python
import argparse
import pathlib

import numpy as np
from colorama import Fore
from tqdm import tqdm

from link_bot_data.tf_dataset_utils import pkl_write_example
from link_bot_pycommon.load_wandb_model import load_model_artifact
from moonshine.numpify import numpify
from moonshine.torch_and_tf_utils import add_batch, remove_batch
from moonshine.torchify import torchify
from state_space_dynamics.meta_udnn import UDNN
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('checkpoint')
    parser.add_argument('threshold', type=float)
    parser.add_argument('--modes', default='val')
    args = parser.parse_args()

    n_low_error = 0
    n_total = 0

    def _remove_meta_mask(mode):
        dataset = TorchDynamicsDataset(args.dataset_dir, mode=mode)
        print(Fore.RED + "Removing meta_mask from {mode}" + Fore.RESET)
        for example in tqdm(dataset):
            example_idx = example['example_idx']
            example.pop('meta_mask', None)
            if 'metadata' in example:
                example['metadata'].pop('meta_mask', None)
            _, full_metadata_filename = pkl_write_example(args.dataset_dir, example, example_idx)
            print(full_metadata_filename)

    def _add_meta_mask(mode):
        nonlocal n_low_error, n_total
        dataset = TorchDynamicsDataset(args.dataset_dir, mode=mode)
        print(Fore.CYAN + f"Adding meta_mask to {mode}" + Fore.RESET)
        for example in tqdm(dataset):
            example_idx = example['example_idx']
            predictions = numpify(remove_batch(model(torchify(add_batch(example)))))
            error = model.scenario.classifier_distance(example, predictions)
            mask = error < args.threshold
            mask = np.logical_and(mask[:-1], mask[1:])
            mask = mask.astype(np.float32)
            n_low_error += mask.sum()
            n_total += mask.size
            mask_padded = np.concatenate([np.zeros(1), mask])  # mask out the first time step
            example['metadata']['meta_mask'] = mask_padded
            # NOTE: upon loading, it copies everything from 'metadata' into the example
            #  but we don't want it to be both in the example and in the metadata,
            #  so remove it from the example before writing
            example.pop('meta_mask', None)
            _, full_metadata_filename = pkl_write_example(args.dataset_dir, example, example_idx)
            print(full_metadata_filename)

    model = load_model_artifact(args.checkpoint, UDNN, project='udnn', version='latest', user='armlab')

    modes = args.modes.split(",")
    for mode in modes:
        _add_meta_mask(mode)

    _remove_meta_mask('test')

    print(f"{n_low_error}/{n_total}={n_low_error / n_total:%} low error")


if __name__ == '__main__':
    main()
