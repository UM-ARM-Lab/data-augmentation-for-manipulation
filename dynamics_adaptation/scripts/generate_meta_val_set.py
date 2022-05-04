#!/usr/bin/env python
import argparse
import pathlib

from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)

    args = parser.parse_args()

    TorchDynamicsDataset(args.dataset_dir, mode='all')

if __name__ == '__main__':
    main()
