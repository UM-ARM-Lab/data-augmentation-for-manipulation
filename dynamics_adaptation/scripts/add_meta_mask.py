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
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset
from state_space_dynamics.udnn_torch import UDNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('checkpoint')
    parser.add_argument('threshold', type=float)

    args = parser.parse_args()

    n_low_error = 0

    def _add_meta_mask(mode):
        nonlocal n_low_error
        dataset = TorchDynamicsDataset(args.dataset_dir, mode=mode)
        print(Fore.CYAN + mode + Fore.RESET)
        for example in tqdm(dataset):
            example_idx = example['example_idx']
            predictions = numpify(remove_batch(model(torchify(add_batch(example)))))
            error = model.scenario.classifier_distance(example, predictions)
            mask = (error < args.threshold).astype(np.float32)
            n_low_error += mask.sum()
            example['metadata']['meta_mask'] = mask
            pkl_write_example(args.dataset_dir, example, example_idx)

    model = load_model_artifact(args.checkpoint, UDNN, project='udnn', version='latest', user='armlab')

    _add_meta_mask('val')
    _add_meta_mask('test')

    print(n_low_error)


if __name__ == '__main__':
    main()
