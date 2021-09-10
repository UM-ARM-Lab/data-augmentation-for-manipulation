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

    tmpdir = f'~\/.tmp_{args.src.name}'
    shutil.copytree(args.src.as_posix(), tmpdir, dirs_exist_ok=True)

    src_pattern = '\/'.join(args.src.parts)

    dest = args.dest / args.src.name
    dest_pattern = '\/'.join(dest.parts)[1:]

    for dirpath, dirnames, filenames in tqdm(os.walk(tmpdir)):
        for filename in filenames:
            if is_valid_filename(filename):
                command = ['sed', '-i', f's/{src_pattern}/{dest_pattern}/g', filename]
                subprocess.run(command, cwd=dirpath, stdout=subprocess.DEVNULL)

    dest_str = dest.as_posix()
    shutil.copytree(tmpdir, dest_str, dirs_exist_ok=True)


if __name__ == '__main__':
    main()
