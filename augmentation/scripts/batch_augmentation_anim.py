#!/usr/bin/env python
import argparse
import logging
import pathlib

import numpy as np
import tensorflow as tf

from arc_utilities import ros_init
from augmentation.augmentation_anim import take_screenshots, make_identifier
from link_bot_data.load_dataset import guess_dataset_loader
from link_bot_pycommon.serialization import my_hdump
from moonshine.filepath_tools import load_hjson
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


@ros_init.with_ros("augmentation_anim")
def main():
    tf.get_logger().setLevel(logging.FATAL)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('name')
    parser.add_argument('aug_hparams', type=pathlib.Path)
    parser.add_argument('data_collection_params', type=pathlib.Path)

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    loader = guess_dataset_loader(dataset_dir)
    scenario = loader.get_scenario()
    params = load_hjson(args.data_collection_params)
    scenario.on_before_data_collection(params)

    figures_info = np.loadtxt(f"{args.name}_figures_info.txt")
    figures_info = np.atleast_2d(figures_info)
    root = pathlib.Path('anims') / args.name

    out_info = {}
    for figure_info_i in figures_info:
        in_idx, aug_seed, tx, ty, tz, r, p, y = figure_info_i

        in_idx = int(in_idx)
        aug_seed = int(aug_seed)

        identifier = make_identifier(in_idx, aug_seed)
        outdir = root / f"{dataset_dir.name}_{identifier}"
        outdir.mkdir(exist_ok=True, parents=True)

        original_filename, output_filename, success = take_screenshots(args.name, outdir, loader, scenario,
                                                                       args.aug_hparams,
                                                                       identifier,
                                                                       in_idx,
                                                                       aug_seed, tx, ty, tz, r, p, y)

        if in_idx not in out_info:
            out_info[in_idx] = {}
            out_info[in_idx]['outputs'] = []
        out_info[in_idx]['original'] = original_filename
        out_info[in_idx]['outputs'].append(output_filename)

    with (root / 'out_info.txt').open("w") as f:
        my_hdump(out_info, f)


if __name__ == '__main__':
    main()
