#!/usr/bin/env python
import argparse
import pathlib
import shutil

import colorama
import hjson


def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("indirs", nargs="*", type=pathlib.Path)
    parser.add_argument("outdir", type=pathlib.Path)
    parser.add_argument("--dry-run", action='store_true')

    args = parser.parse_args()

    args.outdir.mkdir(exist_ok=True)

    file_extension = ".tfrecords"

    if not args.dry_run:
        metadata_filename = 'hparams.hjson'
        path = args.indirs[0] / metadata_filename
        new_path = args.outdir / metadata_filename
        # log this operation in the params!
        hparams = hjson.load(path.open('r'))
        hparams['created_by_merging'] = [str(indir) for indir in args.indirs]
        hjson.dump(hparams, new_path.open('w'), indent=2)
        print(path, '-->', new_path)

    for mode in ['train', 'test', 'val']:
        files = []
        for in_dir in args.indirs:
            mode_indir = in_dir / mode
            tfrecord_files = mode_indir.glob("*" + file_extension)
            files.extend(tfrecord_files)

        traj_idx = 0
        for i, file in enumerate(sorted(files)):
            path = pathlib.Path(file)
            new_filename = index_to_filename(file_extension, traj_idx)
            mode_outdir = args.outdir / mode
            mode_outdir.mkdir(parents=True, exist_ok=True)
            new_path = pathlib.Path(mode_outdir) / new_filename
            traj_idx += 1
            print(path, '-->', new_path)
            if not args.dry_run:
                shutil.copyfile(path, new_path)


def index_to_filename(file_extension, traj_idx):
    new_filename = f"example_{traj_idx:08d}{file_extension}"
    return new_filename


if __name__ == '__main__':
    main()
