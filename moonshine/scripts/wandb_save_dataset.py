#!/usr/bin/env python
import argparse
import pathlib

from link_bot_data.wandb_datasets import wandb_save_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('project', type=str)
    parser.add_argument('--entity', type=str, default='armlab')

    args = parser.parse_args()

    wandb_save_dataset(args.datase_dir, args.project, entity=args.entity)


if __name__ == '__main__':
    main()
