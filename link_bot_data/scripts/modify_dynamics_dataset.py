#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

from arc_utilities import ros_init
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
        keys = ['env', 'extent', 'origin_point', 'res', 'obj0/position', 'obj0/linear_velocity',
                'obj1/position', 'obj1/linear_velocity', 'obj2/position', 'obj2/linear_velocity', 'obj3/position',
                'obj3/linear_velocity', 'obj4/position', 'obj4/linear_velocity', 'obj5/position',
                'obj5/linear_velocity', 'obj6/position', 'obj6/linear_velocity', 'obj7/position',
                'obj7/linear_velocity', 'obj8/position', 'obj8/linear_velocity', 'jaco_arm/primitive_hand/tcp_pos',
                'jaco_arm/primitive_hand/tcp_vel', 'gripper_position', 'num_objs', 'radius', 'height', 'joint_names',
                'time_idx', 'dt', 'jaco_arm/joints_pos', 'is_valid', 'augmented_from', 'filename', 'full_filename',
                'metadata']
        out = {}
        for k in keys:
            if k in example:
                out[k] = example[k]
        yield out

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
