#!/usr/bin/env python
import os
import pathlib
import shutil
import argparse
import subprocess

from tqdm import tqdm


def is_valid_filename(filename):
    return any([
        'json' in filename,
        'txt' in filename,
        'csv' in filename,
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=pathlib.Path)
    parser.add_argument('dest', type=pathlib.Path)

    args = parser.parse_args()

    dest = args.dest / args.src.name
    dest_str = dest.as_posix()
    shutil.copytree(args.src.as_posix(), dest_str, dirs_exist_ok=True)

    src_pattern = '\/'.join(args.src.parts)
    dest_pattern = '\/'.join(dest.parts)[1:]

    for dirpath, dirnames, filenames in tqdm(os.walk(dest_str)):
        for filename in filenames:
            if is_valid_filename(filename):
                command = ['sed', '-i', f's/{src_pattern}/{dest_pattern}/g', filename]
                subprocess.run(command, cwd=dirpath)


if __name__ == '__main__':
    main()
