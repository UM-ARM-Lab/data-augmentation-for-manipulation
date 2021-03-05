#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import colorama

from arc_utilities import ros_init
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_data.modify_dataset import modify_dataset
from link_bot_pycommon.args import my_formatter


@ros_init.with_ros("modify_dynamics_dataset")
def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('suffix', type=str, help='string added to the new dataset name')

    args = parser.parse_args()

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+{args.suffix}"

    def _process_example(dataset: DynamicsDatasetLoader, example: Dict):
        n = [
            'joint56',
            'joint57',
            'joint41',
            'joint42',
            'joint43',
            'joint44',
            'joint45',
            'joint46',
            'joint47',
            'leftgripper',
            'leftgripper2',
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
            'joint7',
            'rightgripper',
            'rightgripper2',
        ]

        example['joint_names'] = 10 * [n]
        yield example

    hparams_update = {}

    dataset = DynamicsDatasetLoader([args.dataset_dir])
    modify_dataset(dataset_dir=args.dataset_dir,
                   dataset=dataset,
                   outdir=outdir,
                   process_example=_process_example,
                   hparams_update=hparams_update)


if __name__ == '__main__':
    main()
