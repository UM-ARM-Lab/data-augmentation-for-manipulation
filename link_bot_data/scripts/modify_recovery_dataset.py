#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import numpy as np

import rospy
from link_bot_data.modify_dataset import modify_dataset
from link_bot_data.recovery_dataset import RecoveryDatasetLoader
from link_bot_pycommon.grid_utils_np import extent_res_to_origin_point


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('suffix', type=str, help='string added to the new dataset name')

    args = parser.parse_args()

    rospy.init_node("modify_recovery_dataset")

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+{args.suffix}"

    def _process_example(dataset, example: Dict):
        example['origin_point'] = extent_res_to_origin_point(example['extent'], example['res']) + np.array([0, 0, 0.0],
                                                                                                           dtype=np.float32)
        yield example

    hparams_update = {
        'env_keys': [
            'env',
            'res',
            'extent',
            'scene_msg',
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
