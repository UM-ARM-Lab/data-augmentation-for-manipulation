#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import colorama

import rospy
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_data.dataset_utils import modify_pad_env
from link_bot_data.modify_dataset import modify_dataset
from link_bot_pycommon.args import my_formatter


def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('x', type=int, help='x')
    parser.add_argument('y', type=int, help='y')
    parser.add_argument('z', type=int, help='z')

    args = parser.parse_args()

    rospy.init_node("modify_dynamics_dataset")

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+{args.x}x{args.y}x{args.z}"

    def _process_example(dataset: ClassifierDatasetLoader, example: Dict):
        example = modify_pad_env(example, args.x, args.y, args.z)
        yield example

    hparams_update = {}

    dataset = ClassifierDatasetLoader([args.dataset_dir], use_gt_rope=False, load_true_states=True)
    modify_dataset(dataset_dir=args.dataset_dir,
                   dataset=dataset,
                   outdir=outdir,
                   process_example=_process_example,
                   hparams_update=hparams_update,
                   slow=False)


if __name__ == '__main__':
    main()
