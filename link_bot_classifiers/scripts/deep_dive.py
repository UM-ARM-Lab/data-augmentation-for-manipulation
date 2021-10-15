import argparse
import logging
import pathlib

import tensorflow as tf

from arc_utilities import ros_init
from link_bot_classifiers.augment_classifier_dataset import augment_classifier_dataset
from link_bot_classifiers.train_test_classifier import train_main, ClassifierEvaluation
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_hjson


@ros_init.with_ros("deep_dive")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('n_augmentations', type=int)
    parser.add_argument('--m', '-m', type=int, default=5)

    args = parser.parse_args()

    tf.get_logger().setLevel(logging.FATAL)

    base_dataset_dir = pathlib.Path("classifier_data/deep_drive_iters0-2")
    mistake_dataset_dir = pathlib.Path("classifier_data/mistake")
    aug_hparams_filename = pathlib.Path("hparams/classifier/aug.hjson")
    aug_hparams = load_hjson(aug_hparams_filename)
    no_aug_hparams_filename = pathlib.Path("hparams/classifier/no_aug.hjson")
    suffix = f"aug-{args.n_augmentations}"
    aug_dataset_dir = base_dataset_dir.parent / f"{base_dataset_dir.name}+{suffix}"
    scenario = get_scenario("dual_arm_rope_sim_val_with_robot_feasibility_checking")

    if not aug_dataset_dir.exists():
        augment_classifier_dataset(dataset_dir=base_dataset_dir,
                                   hparams=aug_hparams,
                                   outdir=aug_dataset_dir,
                                   n_augmentations=args.n_augmentations,
                                   scenario=scenario)

    print(aug_dataset_dir)

    trial_paths = []
    outputs = []
    for i in range(args.m):
        trial_path, _ = train_main(dataset_dirs=[aug_dataset_dir],
                                   model_hparams=no_aug_hparams_filename,
                                   log=f'{base_dataset_dir.name}-{suffix}',
                                   batch_size=16,
                                   seed=i,
                                   epochs=5)
        trial_paths.append(trial_path)
        print(trial_path)

        view = ClassifierEvaluation(dataset_dirs=[mistake_dataset_dir],
                                    checkpoint=trial_path / 'latest_checkpoint',
                                    mode='all',
                                    batch_size=1,
                                    start_at=0)

        outputs_i = []
        for batch_idx, example, predictions in view:
            output = predictions['probabilities'][0, 0, 0].numpy()
            print(output)
            outputs_i.append(output)
        outputs.append(outputs_i)

    print('n_augmentations:', args.n_augmentations)
    print(aug_dataset_dir)
    print(trial_paths)
    for outputs_i in outputs:
        print(outputs_i)


if __name__ == '__main__':
    main()
