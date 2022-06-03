#!/usr/bin/env python
import argparse
import pathlib

import hjson
from colorama import Fore
from tqdm import tqdm

from link_bot_data.tf_dataset_utils import pkl_write_example
from moonshine.filepath_tools import load_hjson
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('--modes', default='test')
    args = parser.parse_args()

    hparams = load_hjson(args.dataset_dir / 'hparams.hjson')
    hparams.pop('meta_mask_threshold', None)
    with (args.dataset_dir / 'hparams.hjson').open('w') as f:
        hjson.dump(hparams, f)

    def _remove_meta_mask(mode):
        dataset = TorchDynamicsDataset(args.dataset_dir, mode=mode)
        print(Fore.RED + f"Removing meta_mask from {mode}" + Fore.RESET)
        for example in tqdm(dataset):
            example_idx = example['metadata']['example_idx']
            example.pop('meta_mask', None)
            if 'metadata' in example:
                example['metadata'].pop('meta_mask', None)
            _, full_metadata_filename = pkl_write_example(args.dataset_dir, example, example_idx)

    modes = args.modes.split(",")
    for mode in modes:
        _remove_meta_mask(mode)


if __name__ == '__main__':
    main()
