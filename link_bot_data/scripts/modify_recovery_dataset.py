#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import colorama

import rospy
from link_bot_data.modify_dataset import modify_dataset
from link_bot_data.recovery_dataset import RecoveryDatasetLoader
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.grid_utils import extent_res_to_origin_point


def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('suffix', type=str, help='string added to the new dataset name')

    args = parser.parse_args()

    rospy.init_node("modify_recovery_dataset")

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+{args.suffix}"

    def _process_example(dataset, example: Dict):
        example['origin_point'] = extent_res_to_origin_point(example['extent'], example['res'])
        yield example

    hparams_update = {
        'env_keys': [
            'env',
            'res',
            'extent',
            'origin',
            'origin_point',
        ]
    }

    dataset = RecoveryDatasetLoader([args.dataset_dir])
    modify_dataset(dataset_dir=args.dataset_dir,
                   dataset=dataset,
                   outdir=outdir,
                   process_example=_process_example,
                   hparams_update=hparams_update)


if __name__ == '__main__':
    main()
