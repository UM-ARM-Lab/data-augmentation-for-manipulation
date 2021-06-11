#!/usr/bin/env python
import argparse
import pathlib

import colorama
import hjson
import tensorflow as tf
from progressbar import progressbar

from arc_utilities import ros_init
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_data.dataset_utils import train_test_split_counts, modify_pad_env, tf_write_example
from link_bot_data.progressbar_widgets import mywidgets
from link_bot_pycommon.pycommon import approx_range_split_counts


@ros_init.with_ros("merge_classifier_datasets")
def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("indirs", nargs="*", type=pathlib.Path)
    parser.add_argument("outdir", type=pathlib.Path)
    parser.add_argument("--take", type=int)
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
        hjson.dump(hparams, new_path.open('w'), indent=2)
        print(path, '-->', new_path)

        # load all the datasets
    datasets = [ClassifierDatasetLoader([d], load_true_states=True) for d in args.indirs]

    # find out the common env size
    max_env_shape = None
    for d in datasets:
        e = next(iter(d.get_datasets(mode='all', take=1)))
        env_shape_i = list(e['env'].shape)
        print(env_shape_i)
        if max_env_shape is None:
            max_env_shape = env_shape_i
        if tf.reduce_any(env_shape_i > max_env_shape):
            max_env_shape = tf.reduce_max([env_shape_i, max_env_shape], axis=0).numpy().tolist()
    print(max_env_shape)

    # how many train/test/val?

    n_datasets = len(args.indirs)
    if args.take:
        n_out = args.take
    else:
        n_out = sum([d.get_datasets(mode="all").size for d in datasets])

    print(f"N Total Examples: {n_out}")

    modes_counts = train_test_split_counts(n_out)

    total_count = 0
    modes = ['train', 'val', 'test']
    for mode, mode_count in zip(modes, modes_counts):
        print(f"n {mode} examples: {mode_count}")
        full_output_directory = args.outdir / mode
        full_output_directory.mkdir(exist_ok=True)

        # how many from each dataset?
        counts_for_each_dataset = approx_range_split_counts(mode_count, n_datasets)
        for d, count_for_dataset in zip(datasets, counts_for_each_dataset):
            mode_dataset = d.get_datasets(mode=mode, take=count_for_dataset, do_not_process=True)
            for e in progressbar(mode_dataset, widgets=mywidgets):
                # deserialize_scene_msg(e)
                # for i in range(10):
                #     d.scenario.plot_environment_rviz(e)
                #     sleep(0.1)
                out_e = modify_pad_env(e, *max_env_shape)
                # input("press enter")
                # for i in range(10):
                #     d.scenario.plot_environment_rviz(out_e)
                #     sleep(0.1)
                # input("press enter")
                if not args.dry_run:
                    tf_write_example(full_output_directory, out_e, total_count)
                total_count += 1


if __name__ == '__main__':
    main()
