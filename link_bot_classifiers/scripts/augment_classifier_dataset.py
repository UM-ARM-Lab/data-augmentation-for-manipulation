#!/usr/bin/env python
import argparse
import logging
import pathlib
from time import time

import tensorflow as tf
from tqdm import tqdm

from arc_utilities import ros_init
from link_bot_classifiers.nn_classifier import NNClassifier
from link_bot_data.dataset_utils import write_example
from link_bot_data.modify_dataset import modify_hparams
from link_bot_data.new_classifier_dataset import NewClassifierDatasetLoader
from link_bot_data.split_dataset import split_dataset
from link_bot_data.visualization import classifier_transition_viz_t
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_hjson
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import batch_examples_dicts, remove_batch

limit_gpu_mem(None)


@ros_init.with_ros("augment_classifier_dataset")
def main():
    tf.get_logger().setLevel(logging.FATAL)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('--n-augmentations', type=int, default=10)
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    suffix = f"aug-{int(time())}"
    dataset_dir = args.dataset_dir
    save_format = 'pkl'

    outdir = dataset_dir.parent / f"{dataset_dir.name}+{suffix}"

    dataset_loader = NewClassifierDatasetLoader([dataset_dir])

    hparams = load_hjson(pathlib.Path("hparams/classifier/aug.hjson"))
    hparams['classifier_dataset_hparams'] = dataset_loader.hparams
    scenario = get_scenario("dual_arm_rope_sim_val_with_robot_feasibility_checking")
    model = NNClassifier(hparams, batch_size=1, scenario=scenario)
    viz_f = classifier_transition_viz_t(metadata={},
                                        state_metadata_keys=dataset_loader.state_metadata_keys,
                                        predicted_state_keys=dataset_loader.predicted_state_keys,
                                        true_state_keys=None)

    def augment(inputs):
        inputs = batch_examples_dicts([inputs])
        if args.visualize:
            scenario.reset_planning_viz()

            inputs_viz = remove_batch(inputs)
            viz_f(scenario, inputs_viz, t=0, idx=0, color='g')
            viz_f(scenario, inputs_viz, t=1, idx=1, color='g')

        for k in range(args.n_augmentations):
            inputs = model.aug.aug_opt(inputs, batch_size=1, time=2)

            if args.visualize:
                inputs_viz = remove_batch(inputs)
                viz_f(scenario, inputs_viz, t=0, idx=2 * k + 2, color='#0000ff88')
                viz_f(scenario, inputs_viz, t=1, idx=2 * k + 3, color='#0000ff88')

            yield inputs

    modify_hparams(dataset_dir, outdir, None)
    dataset = dataset_loader.get_datasets(mode='all', shuffle=False)

    total_count = 0
    for example in tqdm(dataset):
        for out_example in augment(example):
            write_example(outdir, out_example, total_count, save_format)
            total_count += 1

    split_dataset(dataset_dir, val_split=0, test_split=1)


if __name__ == '__main__':
    main()
