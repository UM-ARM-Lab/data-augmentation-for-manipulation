#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import colorama

import rospy
from link_bot_classifiers.classifier_utils import load_generic_model
from link_bot_classifiers.fine_tune_classifier import load_augmentation_configs
from link_bot_classifiers.nn_classifier import NNClassifier
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_data.modify_dataset import modify_dataset
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.grid_utils import extent_res_to_origin_point


def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('suffix', type=str, help='string added to the new dataset name')

    args = parser.parse_args()

    rospy.init_node("modify_classifier_dataset")

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+{args.suffix}"

    def _process_example(dataset: ClassifierDatasetLoader, example: Dict):
        example['origin_point'] = extent_res_to_origin_point(example['extent'], example['res'])
        yield example

    hparams_update = {
        'env_keys': [
            'env',
            'res',
            'extent',
            'origin',
            'origin_point',
        ]
    }

    dataset = ClassifierDatasetLoader([args.dataset_dir], use_gt_rope=False, load_true_states=True)
    modify_dataset(dataset_dir=args.dataset_dir,
                   dataset=dataset,
                   outdir=outdir,
                   process_example=_process_example,
                   hparams_update=hparams_update,
                   slow=False)


if __name__ == '__main__':
    main()
