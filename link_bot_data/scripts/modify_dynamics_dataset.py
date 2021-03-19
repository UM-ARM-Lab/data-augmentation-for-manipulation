#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import colorama

from arc_utilities import ros_init
from link_bot_data.dataset_utils import modify_pad_env, deserialize_scene_msg
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_data.modify_dataset import modify_dataset
from link_bot_pycommon.args import my_formatter


@ros_init.with_ros("modify_dynamics_dataset")
def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('suffix', type=str, help='string added to the new dataset name')
    parser.add_argument('x', type=int, help='x')
    parser.add_argument('y', type=int, help='y')
    parser.add_argument('z', type=int, help='z')

    args = parser.parse_args()

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+{args.suffix}"

    def _process_example(dataset: DynamicsDatasetLoader, example: Dict):
        deserialize_scene_msg(example)
        example = modify_pad_env(example, args.x, args.y, args.z)
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
