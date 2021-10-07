#!/usr/bin/env python
import argparse
import pathlib

from link_bot_data.modify_classifier_dataset import modify_classifier_dataset
from sdf_tools import utils_3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')

    args = parser.parse_args()
    suffix = 'sdf'

    def add_sdf_and_grad(_, example):
        sdf, sdf_grad = utils_3d.compute_sdf_and_gradient(example['env'], example['res'], example['origin_point'])
        example['sdf'] = sdf
        example['sdf_grad'] = sdf_grad
        return example

    modify_classifier_dataset(args.dataset_dir, suffix, add_sdf_and_grad)


if __name__ == '__main__':
    main()
