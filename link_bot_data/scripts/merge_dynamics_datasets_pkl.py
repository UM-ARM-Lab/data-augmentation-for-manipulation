#!/usr/bin/env python
import argparse
import pathlib

import colorama

from link_bot_data.merge_dynamics_datasets_pkl import merge_dynamics_datasets_pkl


def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("indirs", nargs="*", type=pathlib.Path)
    parser.add_argument("outdir", type=pathlib.Path)

    args = parser.parse_args()

    merge_dynamics_datasets_pkl(**vars(args))


if __name__ == '__main__':
    main()
