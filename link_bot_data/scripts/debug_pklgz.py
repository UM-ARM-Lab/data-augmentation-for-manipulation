#!/usr/bin/env python
import argparse
import pathlib
from pprint import pprint

from colorama import Fore

from link_bot_pycommon.serialization import load_gzipped_pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', type=pathlib.Path)

    args = parser.parse_args()

    for e in args.indir.iterdir():
        if e.is_file() and 'pkl.gz' in e.name:
            print(Fore.GREEN + e.as_posix() + Fore.RESET)
            d = load_gzipped_pickle(e)
            pprint(d)
            return


if __name__ == '__main__':
    main()
