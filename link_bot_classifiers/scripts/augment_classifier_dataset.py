#!/usr/bin/env python
import argparse
import pathlib
from time import time

from tqdm import tqdm

from arc_utilities import ros_init
from link_bot_classifiers.nn_classifier import NNClassifier
from link_bot_data.dataset_utils import write_example
from link_bot_data.modify_dataset import modify_hparams
from link_bot_data.new_classifier_dataset import NewClassifierDatasetLoader
from link_bot_data.split_dataset import split_dataset
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_hjson
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import batch_examples_dicts

limit_gpu_mem(None)


@ros_init.with_ros("augment_classifier_dataset")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('--n-augmentations', type=int, default=10)

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

    def augment(inputs):
        inputs = batch_examples_dicts([inputs])
        for _ in range(args.n_augmentations):
            inputs = model.aug.augmentation_optimization(inputs, batch_size=1, time=2)
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
