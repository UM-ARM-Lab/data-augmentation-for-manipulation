#!/usr/bin/env python
import argparse
import pathlib

from link_bot_data.modify_classifier_dataset import modify_classifier_dataset
from sdf_tools import utils_3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, help='dataset directory', nargs='+')

    args = parser.parse_args()
    suffix = 'sdf'

    def add_sdf_and_grad(_, example):
        sdf, sdf_grad = utils_3d.compute_sdf_and_gradient(example['env'], example['res'], example['origin_point'])
        example['sdf'] = sdf
        example['sdf_grad'] = sdf_grad
        yield example

    hparams_update = {'env_keys': [
        'env',
        'extent',
        'origin',
        'origin_point',
        'res',
        'scene_msg',
        'sdf',
        'sdf_grad',
    ]}
    for dataset_dir in args.dataset_dirs:
        modify_classifier_dataset(dataset_dir, suffix, add_sdf_and_grad, hparams_update=hparams_update)


if __name__ == '__main__':
    main()
