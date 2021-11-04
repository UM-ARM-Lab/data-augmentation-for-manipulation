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


def pos_to_vel(pos):
    vel = pos[1:] - pos[:-1]
    vel = np.pad(vel, [[1, 0], [0, 0], [0, 0]])
    return vel


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
        num_objs = example['num_objs'][0, 0]  # assumed fixed across time
        robot_pos = example[f'{ARM_HAND_NAME}/tcp_pos']
        robot_vel = pos_to_vel(robot_pos)
        example[f"{ARM_HAND_NAME}/tcp_vel"] = robot_vel
        for obj_idx in range(num_objs):
            obj_pos = example[f'{ARM_HAND_NAME}/position']
            obj_vel = pos_to_vel(obj_pos)
            example[f"obj{obj_idx}/linear_velocity"] = obj_vel
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
        split_dataset_via_files(args.dataset_dir, 'hjson')


if __name__ == '__main__':
    main()
