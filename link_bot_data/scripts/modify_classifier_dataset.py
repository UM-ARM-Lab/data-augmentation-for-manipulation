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
from link_bot_data.dataset_utils import add_new
from link_bot_data.modify_dataset import modify_dataset
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.moonshine_utils import add_batch, repeat


def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('suffix', type=str, help='string added to the new dataset name')

    args = parser.parse_args()

    rospy.init_node("modify_classifier_dataset")

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+{args.suffix}"

    s = get_scenario('dual_arm_rope_sim_val_with_robot_feasibility_checking')
    p = pathlib.Path("cl_trials/val_car_feasible_1614981888/March_05_18-57-54_4b65490ac1/best_checkpoint/")
    classifier = load_generic_model(p, scenario=s)
    net = classifier.net
    net.augmentation_3d = NNClassifier.fitted_env_augmentation
    net.is_env_augmentation_valid = NNClassifier.is_env_augmentation_valid_discrete

    pretransfer_config_gen = load_augmentation_configs(pathlib.Path("pretransfer_initial_configs/long_hook/"))

    def _process_example(dataset: ClassifierDatasetLoader, example: Dict):
        example['swept_']
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
