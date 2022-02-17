#!/usr/bin/env python
import argparse
import pathlib

import colorama
import hjson
from progressbar import progressbar

from arc_utilities import ros_init
from link_bot_data.dataset_utils import train_test_split_counts, DEFAULT_TEST_SPLIT, DEFAULT_VAL_SPLIT
from link_bot_data.tf_dataset_utils import write_example
from link_bot_data.load_dataset import get_classifier_dataset_loader, guess_dataset_format
from link_bot_data.progressbar_widgets import mywidgets
from link_bot_data.split_dataset import split_dataset
from moonshine.filepath_tools import load_hjson


@ros_init.with_ros("merge_classifier_datasets")
def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("indirs", nargs="*", type=pathlib.Path)
    parser.add_argument("outdir", type=pathlib.Path)
    parser.add_argument("--test-split", type=float, default=DEFAULT_TEST_SPLIT)
    parser.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--take", type=int)
    parser.add_argument("--save-format", choices=['pkl', 'tfrecord'], default='pkl')
    parser.add_argument("--dry-run", action='store_true')

    args = parser.parse_args()

    args.outdir.mkdir(exist_ok=True)

    if not args.dry_run:
        metadata_filename = 'hparams.hjson'
        path = args.indirs[0] / metadata_filename
        new_path = args.outdir / metadata_filename
        # log this operation in the params!
        hparams = hjson.load(path.open('r'))
        hparams['created_by_merging'] = [str(indir) for indir in args.indirs]
        with new_path.open('w') as f:
            hjson.dump(hparams, f, indent=2)
        print(path, '-->', new_path)

        # load all the datasets
    datasets = [get_classifier_dataset_loader([d], load_true_states=True) for d in args.indirs]

    # how many train/test/val?
    n_datasets = len(args.indirs)
    if args.take:
        n_out = args.take
    else:
        n_out = sum([len(d.get_datasets(mode="all")) for d in datasets])

    print(f"N Total Examples: {n_out}")

    modes_counts = train_test_split_counts(n_out, val_split=0, test_split=0)

    combined_manual_transforms = {}
    total_count = 0
    has_manual_transforms = True
    for d in datasets:
        manual_transforms_filename = d.dataset_dirs[0] / 'manual_transforms.hjson'
        if not manual_transforms_filename.exists():
            has_manual_transforms = False
            continue
        manual_transforms = load_hjson(manual_transforms_filename)
        for transforms in manual_transforms.values():
            new_example_name = f'example_{total_count:08d}'
            combined_manual_transforms[new_example_name] = transforms
            total_count += 1
    if has_manual_transforms:
        combined_manual_transforms_filename = args.outdir / 'manual_transforms.hjson'
        with combined_manual_transforms_filename.open("w") as f:
            hjson.dump(combined_manual_transforms, f)

    total_count = 0
    modes = ['train', 'val', 'test']
    input_format = guess_dataset_format(args.indirs[0])
    for mode, mode_count in zip(modes, modes_counts):
        print(f"n {mode} examples: {mode_count}")
        if input_format == 'tfrecord':
            full_output_directory = args.outdir / mode
            full_output_directory.mkdir(exist_ok=True)
        else:
            full_output_directory = args.outdir

        # how many from each dataset?
        # counts_for_each_dataset = approx_range_split_counts(mode_count, n_datasets)
        # for d, count_for_dataset in zip(datasets, counts_for_each_dataset):
        for d in datasets:
            mode_dataset = d.get_datasets(mode=mode, do_not_process=True)
            for e in progressbar(mode_dataset, widgets=mywidgets):
                if not args.dry_run:
                    write_example(full_output_directory, e, total_count, save_format='pkl')
                total_count += 1

    split_dataset(full_output_directory, val_split=args.val_split, test_split=args.test_split)


if __name__ == '__main__':
    main()
