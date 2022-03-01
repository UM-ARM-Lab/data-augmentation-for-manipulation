#!/usr/bin/env python

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)

    args = parser.parse_args()

    modes = ['train', 'val', 'test']
    for mode in modes:
        dataset = TorchDynamicsDataset(args.dataset_dir, mode=mode)
        weights_for_mode = []
        for e in dataset:
            weight = e['weight']
            weights_for_mode.extend(weight.tolist())

        plt.figure()
        c, _, bars = plt.hist(weights_for_mode)
        plt.bar_label(bars)
        plt.title(f"Weights for mode = {mode}")
        plt.xlabel("weight")

    plt.show()


if __name__ == '__main__':
    main()
