#!/usr/bin/env python
import argparse
import logging
import pathlib

import tensorflow as tf

from arc_utilities import ros_init
from augmentation.augmentation_anim import take_screenshots, make_identifier
from link_bot_data.load_dataset import guess_dataset_loader
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


@ros_init.with_ros("augmentation_anim_dmd")
def main():
    tf.get_logger().setLevel(logging.FATAL)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('aug_hparams', type=pathlib.Path)
    parser.add_argument('in_idx', type=int)
    parser.add_argument('aug_seed', type=int)
    parser.add_argument('tx', type=float)
    parser.add_argument('ty', type=float)
    parser.add_argument('tz', type=float)
    parser.add_argument('r', type=float)
    parser.add_argument('p', type=float)
    parser.add_argument('y', type=float)

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    loader = guess_dataset_loader(dataset_dir)
    scenario = loader.get_scenario()

    outdir = pathlib.Path('anims') / 'dmd'
    outdir.mkdir(exist_ok=True, parents=True)

    identifier = "dmd_" + make_identifier(args.in_idx, args.aug_seed)
    take_screenshots(name='rope', outdir=outdir, loader=loader, scenario=scenario, identifier=identifier,
                     hparams_filename=args.aug_hparams,
                     in_idx=args.in_idx, aug_seed=args.aug_seed,
                     tx=args.tx, ty=args.ty, tz=args.tz, r=args.r, p=args.p, y=args.y)


if __name__ == '__main__':
    main()
