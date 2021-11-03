#!/usr/bin/env python
import argparse
import pathlib
import shutil

import colorama
import hjson

from link_bot_data.dataset_utils import index_to_filename
from link_bot_data.split_dataset import split_dataset


def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("indirs", nargs="*", type=pathlib.Path)
    parser.add_argument("outdir", type=pathlib.Path)
    parser.add_argument("--dry-run", action='store_true')

    args = parser.parse_args()

    args.outdir.mkdir(exist_ok=True)

    if not args.dry_run:
        hparams_filename = 'hparams.hjson'
        path = args.indirs[0] / hparams_filename
        new_hparams_filename = args.outdir / hparams_filename
        # log this operation in the params!
        hparams = hjson.load(path.open('r'))
        hparams['created_by_merging'] = [str(indir) for indir in args.indirs]
        with new_hparams_filename.open('w') as new_hparams_file:
            hjson.dump(hparams, new_hparams_file, indent=2)
        print(path, '-->', new_hparams_filename)

    pkl_files = []
    pkl_gz_files = []
    for in_dir in args.indirs:
        pkl_files.extend(in_dir.glob("*.pkl"))
        pkl_gz_files.extend(in_dir.glob("*.pkl.gz"))

    traj_idx = 0
    for i, file in enumerate(sorted(pkl_files)):
        path = pathlib.Path(file)
        new_pkl_filename = index_to_filename('.pkl', traj_idx)
        new_pkl_path = pathlib.Path(args.outdir) / new_pkl_filename
        traj_idx += 1
        print(path, '-->', new_pkl_path)
        if not args.dry_run:
            shutil.copyfile(path, new_pkl_path)

    traj_idx = 0
    for i, file in enumerate(sorted(pkl_gz_files)):
        path = pathlib.Path(file)
        new_pkl_gz_filename = index_to_filename('.pkl.gz', traj_idx)
        new_pkl_gz_path = pathlib.Path(args.outdir) / new_pkl_gz_filename
        traj_idx += 1
        print(path, '-->', new_pkl_gz_path)
        if not args.dry_run:
            shutil.copyfile(path, new_pkl_gz_path)

    if not args.dry_run:
        split_dataset(args.outdir)


if __name__ == '__main__':
    main()
