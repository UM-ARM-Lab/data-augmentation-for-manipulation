#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt

from cylinders_simple_demo.utils.my_torch_dataset import MyTorchDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('mode', type=str, choices=['train', 'val', 'test'])

    args = parser.parse_args()

    dataset = MyTorchDataset(args.dataset_dir, mode=args.mode)
    s = dataset.get_scenario()

    for example in dataset:
        anim = s.example_to_animation(example)
        # anim.save()
        plt.show()


if __name__ == '__main__':
    main()
