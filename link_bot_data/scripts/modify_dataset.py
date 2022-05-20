#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import numpy as np

from arc_utilities import ros_init
from link_bot_data.tf_dataset_utils import pkl_write_example
from moonshine.my_torch_dataset import MyTorchDataset


@ros_init.with_ros("modify_dataset")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('suffix', type=str, help='string added to the new dataset name')

    args = parser.parse_args()

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+{args.suffix}"

    def _process_example(dataset, example: Dict):
        example['time_idx'] = example['time_idx'].astype(np.float32)
        yield example

    dataset = MyTorchDataset(args.dataset_dir, mode='all', no_update_with_metadata=True)

    for example in dataset:
        new_example = _process_example(dataset, example)
        pkl_write_example(outdir, new_example, new_example['example_idx'])


if __name__ == '__main__':
    main()
