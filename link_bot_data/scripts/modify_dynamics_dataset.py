#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import numpy as np

from arc_utilities import ros_init
from dm_envs.planar_pushing_task import ARM_HAND_NAME
from learn_invariance.new_dynamics_dataset import NewDynamicsDatasetLoader
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_data.modify_dataset import modify_dataset, modify_dataset2
from link_bot_data.split_dataset import split_dataset_via_files



@ros_init.with_ros("modify_dynamics_dataset")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('suffix', type=str, help='string added to the new dataset name')
    parser.add_argument('--save-format', type=str, choices=['pkl', 'tfrecord'], default='pkl')

    args = parser.parse_args()

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+{args.suffix}"

    s = None

    def _process_example(dataset, example: Dict):
        example = s.propnet_add_vel(example)
        yield example

    hparams_update = {}

    if False:
        dataset = DynamicsDatasetLoader([args.dataset_dir])
        modify_dataset(dataset_dir=args.dataset_dir,
                       dataset=dataset,
                       outdir=outdir,
                       process_example=_process_example,
                       hparams_update=hparams_update,
                       save_format=args.save_format)
        s = dataset.get_scenario()
    else:
        dataset = NewDynamicsDatasetLoader([args.dataset_dir])
        s = dataset.get_scenario()
        modify_dataset2(dataset_dir=args.dataset_dir,
                        dataset=dataset,
                        outdir=outdir,
                        process_example=_process_example,
                        hparams_update=hparams_update,
                        save_format=args.save_format)
        split_dataset_via_files(args.dataset_dir, 'pkl')


if __name__ == '__main__':
    main()
