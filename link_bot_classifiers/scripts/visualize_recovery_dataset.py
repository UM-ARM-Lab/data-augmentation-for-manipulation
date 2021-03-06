#!/usr/bin/env python
import argparse
import pathlib

import colorama
import numpy as np
from progressbar import progressbar

from arc_utilities import ros_init
from link_bot_data.dataset_utils import pprint_example
from link_bot_data.tf_dataset_utils import deserialize_scene_msg
from link_bot_data.progressbar_widgets import mywidgets
from link_bot_data.recovery_dataset import RecoveryDatasetLoader
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(1)


@ros_init.with_ros("visualize_recovery_dataset")
def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('--type', choices=['best_to_worst', 'in_order', 'stats'], default='best_to_worst')
    parser.add_argument('--mode', choices=['train', 'val', 'test', 'all'], default='train')

    args = parser.parse_args()

    dataset = RecoveryDatasetLoader(args.dataset_dirs)
    if args.type == 'stats':
        stats(args, dataset)
    else:
        if args.type == 'best_to_worst':
            tf_dataset = dataset.get_datasets(mode=args.mode, sort=True)
        else:
            tf_dataset = dataset.get_datasets(mode=args.mode)

        example = tf_dataset.get_element(0)
        print("Example:")
        pprint_example(example)

        for example in progressbar(tf_dataset, widgets=mywidgets):
            # if not is_stuck(example):
            #     print("found a not-stuck example")
            #     dataset.anim_rviz(example)
            # print(example['recovery_probability'])
            deserialize_scene_msg(example)
            dataset.anim_rviz(example)


def stats(args, dataset):
    recovery_probabilities = []
    batch_size = 512
    tf_dataset = dataset.batch(batch_size, drop_remainder=True)
    for example in tf_dataset:
        batch = example['recovery_probability'][:, 1].numpy().tolist()
        recovery_probabilities.extend(batch)

    print(f"Num examples: {len(recovery_probabilities)}")
    print(f'mean: {np.mean(recovery_probabilities):.5f}')
    print(f'median: {np.median(recovery_probabilities):.5f}')
    print(f'min: {np.min(recovery_probabilities):.5f}')
    print(f'max: {np.max(recovery_probabilities):.5f}')


if __name__ == '__main__':
    main()
