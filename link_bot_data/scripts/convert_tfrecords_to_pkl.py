#!/usr/bin/env python
import argparse
import pathlib

from colorama import Fore
from tqdm import tqdm

from arc_utilities import ros_init
from link_bot_data.tf_dataset_utils import write_example
from link_bot_data.load_dataset import get_dynamics_dataset_loader
from link_bot_data.split_dataset import split_dataset_via_files


@ros_init.with_ros("modify_dataset")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('--save-format', type=str, choices=['pkl', 'tfrecord'], default='pkl')

    args = parser.parse_args()

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+pkl"
    outdir.mkdir()

    loader = get_dynamics_dataset_loader(args.dataset_dir)

    total_count = 0
    for mode in ['train', 'test', 'val']:
        dataset = loader.get_datasets(mode=mode)
        for i, example in enumerate(tqdm(dataset)):
            write_example(outdir, example, total_count, save_format='pkl')
            total_count += 1
    print(Fore.GREEN + f"Modified {total_count} examples")

    split_dataset_via_files(outdir, 'pkl')


if __name__ == '__main__':
    main()
