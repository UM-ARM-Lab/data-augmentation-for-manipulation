#!/usr/bin/env python
import argparse
import pathlib
import shutil

import colorama
import hjson

from moonshine.filepath_tools import load_hjson


def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", type=pathlib.Path)
    parser.add_argument("outdir", type=pathlib.Path)
    parser.add_argument("--dry-run", action='store_true')

    args = parser.parse_args()

    args.outdir.mkdir(exist_ok=True)

    file_extension = ".pkl.gz"
    metadata_filename = 'metadata.json'

    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        path = args.indir / metadata_filename
        new_path = args.outdir / metadata_filename
        # log this operation in the params!
        hparams = load_hjson(path)
        hparams['created_by_merging'] = hparams.get('created_by_merging', []) + [args.indir.as_posix()]
        hjson.dump(hparams, new_path.open('w'), indent=2)
        print(path, '-->', new_path)

    paths_to_merge = args.indir.glob("*" + file_extension)

    existing_paths = args.outdir.glob("*" + file_extension)
    traj_idx = 0
    for path in existing_paths:
        traj_idx += 1

    for i, path in enumerate(sorted(paths_to_merge)):
        new_filename = index_to_filename(file_extension, traj_idx)
        new_path = args.outdir / new_filename
        traj_idx += 1
        print(path, '-->', new_path)
        if new_path.exists():
            print(f"refusing to override existing file {new_path.as_posix()}")
        else:
            if not args.dry_run:
                shutil.copyfile(path, new_path)


def index_to_filename(file_extension, traj_idx):
    new_filename = f"{traj_idx}_metrics{file_extension}"
    return new_filename


if __name__ == '__main__':
    main()
