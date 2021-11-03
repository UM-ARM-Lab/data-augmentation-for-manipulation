#!/usr/bin/env python
import argparse
import pathlib

import colorama

from link_bot_data.merge_pkls import merge_pkls


def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("indirs", nargs="*", type=pathlib.Path)
    parser.add_argument("outdir", type=pathlib.Path)
    parser.add_argument("--dry-run", action='store_true')

    args = parser.parse_args()

    merge_pkls(**vars(args))


if __name__ == '__main__':
    main()
