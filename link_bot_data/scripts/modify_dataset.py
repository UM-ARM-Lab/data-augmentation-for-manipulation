#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

from arc_utilities import ros_init
from link_bot_data.modify_dataset import modify_dataset2
from link_bot_data.new_base_dataset import NewBaseDatasetLoader
from link_bot_data.split_dataset import split_dataset_via_files


@ros_init.with_ros("modify_dataset")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('suffix', type=str, help='string added to the new dataset name')
    parser.add_argument('--save-format', type=str, choices=['pkl', 'tfrecord'], default='pkl')

    args = parser.parse_args()

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+{args.suffix}"

    def _process_example(dataset, example: Dict):
        keys = ['error', 'transform']
        out = {}
        for k in keys:
            if k in example:
                out[k] = example[k]
        yield out

    hparams_update = {}

    dataset = NewBaseDatasetLoader([args.dataset_dir])
    modify_dataset2(dataset_dir=args.dataset_dir,
                    dataset=dataset,
                    outdir=outdir,
                    process_example=_process_example,
                    hparams_update=hparams_update,
                    save_format=args.save_format)
    split_dataset_via_files(args.dataset_dir, 'pkl')


if __name__ == '__main__':
    main()