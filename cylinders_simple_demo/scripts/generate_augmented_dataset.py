#!/usr/bin/env python
import argparse
import logging
import pathlib
from typing import Optional, Dict

import hjson as hjson
import numpy as np
import tensorflow as tf

from cylinders_simple_demo.augment_cylinders_dataset import augment_dataset_from_loader
from cylinders_simple_demo.cylinders_dynamics_dataset import CylindersDynamicsDatasetLoader
from cylinders_simple_demo.cylinders_scenario import CylindersScenario
from cylinders_simple_demo.utils import load_hjson
from link_bot_data.visualization import init_viz_env, dynamics_viz_t
from merrrt_visualization.rviz_animation_controller import RvizAnimation
from moonshine.numpify import numpify


def rm_tree(path):
    path = pathlib.Path(path)
    for child in path.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    path.rmdir()


def augment_dynamics_dataset(dataset_dir: pathlib.Path,
                             mode: str,
                             hparams: Dict,
                             outdir: pathlib.Path,
                             n_augmentations: int,
                             take: Optional[int] = None,
                             visualize: bool = False,
                             batch_size: int = 32,
                             use_torch: bool = True,
                             save_format='pkl'):
    loader = CylindersDynamicsDatasetLoader([dataset_dir])
    scenario = CylindersScenario()

    def viz_f(_scenario, example, **kwargs):
        example = numpify(example)
        state_keys = list(filter(lambda k: k in example, loader.state_keys))
        anim = RvizAnimation(_scenario,
                             n_time_steps=example['time_idx'].size,
                             init_funcs=[
                                 init_viz_env
                             ],
                             t_funcs=[
                                 init_viz_env,
                                 dynamics_viz_t(metadata={},
                                                label='aug',
                                                state_metadata_keys=loader.state_metadata_keys,
                                                state_keys=state_keys,
                                                action_keys=loader.action_keys),
                             ])
        anim.play(example)

    debug_state_keys = loader.state_keys
    outdir = augment_dataset_from_loader(loader,
                                         viz_f,
                                         dataset_dir,
                                         mode,
                                         take,
                                         hparams,
                                         outdir,
                                         n_augmentations,
                                         debug_state_keys,
                                         scenario,
                                         visualize,
                                         batch_size,
                                         use_torch)

    return outdir


def limit_gpu_mem(gigs: Optional[float]):
    gpus = tf.config.list_physical_devices('GPU')
    gpu = gpus[0]
    tf.config.experimental.set_memory_growth(gpu, True)
    if gigs is not None:
        config = [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * gigs)]
        tf.config.set_logical_device_configuration(gpu, config)


def main():
    limit_gpu_mem(None)

    np.set_printoptions(suppress=True, precision=4)

    tf.get_logger().setLevel(logging.FATAL)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('hparams', type=pathlib.Path, help='hyper-parameters for augmentation')
    parser.add_argument('outdir', type=pathlib.Path, help='output directory')
    parser.add_argument('--n-augmentations', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--take', type=int)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--use-torch', action='store_true')

    args = parser.parse_args()

    dataset_dir = args.dataset_dir

    hparams = load_hjson(args.hparams)
    hparams['n_augmentations'] = args.n_augmentations

    if args.outdir.exists():
        rm_tree(args.outdir)

    augment_dynamics_dataset(dataset_dir=dataset_dir,
                             hparams=hparams,
                             mode=args.mode,
                             take=args.take,
                             outdir=args.outdir,
                             n_augmentations=args.n_augmentations,
                             use_torch=args.use_torch,
                             visualize=args.visualize,
                             batch_size=args.batch_size)


if __name__ == '__main__':
    main()
