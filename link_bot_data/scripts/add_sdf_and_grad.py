#!/usr/bin/env python
import argparse
import pathlib
import shutil

from arc_utilities import ros_init
from link_bot_data.modify_classifier_dataset import modify_classifier_dataset
from sdf_tools import utils_3d


@ros_init.with_ros("add_sdf_and_grad")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, help='dataset directory', nargs='+')

    args = parser.parse_args()
    suffix = 'sdf'

    def add_sdf_and_grad(_, example):
        if 'sdf' not in example or 'sdf_grad' not in example:
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
        outdir = modify_classifier_dataset(dataset_dir, suffix, add_sdf_and_grad, hparams_update=hparams_update)
        dataset_dir_bak = dataset_dir.parent / (dataset_dir.name + '.bak')
        shutil.copytree(dataset_dir, dataset_dir_bak, dirs_exist_ok=True)
        shutil.copytree(outdir, dataset_dir, dirs_exist_ok=True)


if __name__ == '__main__':
    main()
