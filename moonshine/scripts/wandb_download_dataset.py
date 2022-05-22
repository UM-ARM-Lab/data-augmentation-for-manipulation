#!/usr/bin/env python
import argparse
import pathlib

from link_bot_data.wandb_datasets import wandb_download_dataset_to


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('project', type=str)
    parser.add_argument('dataset_name', type=str)
    parser.add_argument('outdir', type=pathlib.Path)
    parser.add_argument('--entity', type=str, default='armlab')
    parser.add_argument('--version', type=str, default='latest')

    args = parser.parse_args()

    full_outdir = wandb_download_dataset_to(args.entity, args.project, args.dataset_name, args.version, args.outdir)
    print(full_outdir)


if __name__ == '__main__':
    main()
