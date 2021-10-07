#!/usr/bin/env python
import argparse
import pathlib
from time import time

import tensorflow as tf

from arc_utilities import ros_init
from link_bot_classifiers.add_augmentation_configs import make_add_augmentation_env_func
from link_bot_classifiers.nn_classifier import NNClassifier
from link_bot_data.modify_dataset import modify_dataset2
from link_bot_data.new_classifier_dataset import NewClassifierDatasetLoader
from link_bot_data.split_dataset import split_dataset
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_hjson
from moonshine.moonshine_utils import add_batch, batch_examples_dicts


@ros_init.with_ros("augment_classifier_dataset")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')

    args = parser.parse_args()

    suffix = f"aug-{int(time())}"
    dataset_dir = args.dataset_dir
    save_format = 'pkl'
    augmentation_config_dir = pathlib.Path("/media/shared/pretransfer_initial_configs/car3")

    outdir = dataset_dir.parent / f"{dataset_dir.name}+{suffix}"

    dataset_loader = NewClassifierDatasetLoader([dataset_dir])

    hparams = load_hjson(pathlib.Path("hparams/classifier/aug.hjson"))
    hparams['classifier_dataset_hparams'] = dataset_loader.hparams
    scenario = get_scenario("dual_arm_rope_sim_val_with_robot_feasibility_checking")
    model = NNClassifier(hparams, batch_size=1, scenario=scenario)

    add_augmentation_env_func = make_add_augmentation_env_func(augmentation_config_dir, batch_size=1)

    def augment(_, inputs):
        inputs = batch_examples_dicts([inputs])
        inputs, local_env, local_origin_point = model.aug.augmentation_optimization(inputs, batch_size=1, time=2)
        return inputs

    modify_dataset2(dataset_dir=dataset_dir,
                    dataset=dataset_loader,
                    outdir=outdir,
                    process_example=augment,
                    hparams_update={},
                    save_format=save_format)

    split_dataset(dataset_dir, val_split=0, test_split=1)


if __name__ == '__main__':
    main()
