#!/usr/bin/env python

import argparse
import pathlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name1')
    parser.add_argument('name2')
    args = parser.parse_args()

    root = pathlib.Path("results")
    for d1 in root.iterdir():
        if d1.is_dir():
            if args.name1 in d1.name:
                for d2 in d1.iterdir():
                    if d2.is_dir():
                        if args.name2 in d2.name:
                            print(d2.as_posix())


if __name__ == '__main__':
    main()
