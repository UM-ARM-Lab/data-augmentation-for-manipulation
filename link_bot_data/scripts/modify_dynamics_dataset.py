#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

from arc_utilities import ros_init
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_data.modify_dataset import modify_dataset
from link_bot_pycommon.grid_utils import extent_res_to_origin_point


@ros_init.with_ros("modify_dynamics_dataset")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('suffix', type=str, help='string added to the new dataset name')

    args = parser.parse_args()

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+{args.suffix}"

    def _process_example(dataset: DynamicsDatasetLoader, example: Dict):
        example['origin_point'] = extent_res_to_origin_point(example['extent'], example['res'])
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
