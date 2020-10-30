#!/usr/bin/env python
import argparse
import json
import pathlib
from time import time

import colorama
import numpy as np
import tensorflow as tf
from colorama import Fore

import rospy
import state_space_dynamics
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_data.link_bot_dataset_utils import batch_tf_dataset
from link_bot_pycommon.rviz_animation_controller import RvizSimpleStepper, RvizAnimationController
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import remove_batch, numpify
from my_cfm.cfm import CFM
from shape_completion_training.model.filepath_tools import load_trial
from shape_completion_training.model_runner import ModelRunner
from state_space_dynamics import train_test

limit_gpu_mem(8)


def train_main(args):
    dataset_dirs = args.dataset_dirs
    checkpoint = args.checkpoint
    epochs = args.epochs
    trial_path, params = load_trial(checkpoint.parent.absolute())
    now = str(time())
    trial_path = trial_path.parent / (trial_path.name + '-observer-' + now)
    trial_path.mkdir(parents=True)
    batch_size = params['batch_size']
    params['encoder_trainable'] = False
    params['use_observation_feature_loss'] = True
    params['use_cfm_loss'] = False
    out_hparams_filename = trial_path / 'params.json'
    out_params_str = json.dumps(params)
    with out_hparams_filename.open("w") as out_hparams_file:
        out_hparams_file.write(out_params_str)

    train_dataset = DynamicsDataset(dataset_dirs)
    val_dataset = DynamicsDataset(dataset_dirs)

    model_class = state_space_dynamics.get_model(params['model_class'])
    model = model_class(hparams=params, batch_size=batch_size, scenario=train_dataset.scenario)

    seed = 0

    runner = ModelRunner(model=model,
                         training=True,
                         params=params,
                         checkpoint=checkpoint,
                         batch_metadata=train_dataset.batch_metadata,
                         trial_path=trial_path)

    train_tf_dataset, val_tf_dataset = train_test.setup_datasets(model_hparams=params,
                                                                 batch_size=batch_size,
                                                                 seed=seed,
                                                                 train_dataset=train_dataset,
                                                                 val_dataset=val_dataset)

    runner.train(train_tf_dataset, val_tf_dataset, num_epochs=epochs)

    return trial_path


def viz_main(args):
    dataset_dirs = args.dataset_dirs
    checkpoint = args.checkpoint

    trial_path, params = load_trial(checkpoint.parent.absolute())

    dataset = DynamicsDataset(dataset_dirs)
    scenario = dataset.scenario

    tf_dataset = dataset.get_datasets(mode='val')
    tf_dataset = batch_tf_dataset(tf_dataset, batch_size=1, drop_remainder=True)

    model = CFM(hparams=params, batch_size=1, scenario=scenario)
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, args.checkpoint, max_to_keep=1)
    status = ckpt.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        print(Fore.CYAN + "Restored from {}".format(manager.latest_checkpoint))
        status.assert_existing_objects_matched()
    else:
        raise RuntimeError("Failed to restore!!!")

    for example_idx, example in enumerate(tf_dataset):
        stepper = RvizAnimationController(n_time_steps=dataset.steps_per_traj)
        for t in range(dataset.steps_per_traj):
            output = model(model.preprocess_no_gradient(example, training=False))

            actual_t = numpify(remove_batch(scenario.index_time_batched_predicted(example, t)))
            action_t = numpify(remove_batch(scenario.index_time_batched_predicted(example, t)))
            scenario.plot_state_rviz(actual_t, label='actual', color='red')
            scenario.plot_action_rviz(actual_t, action_t, color='gray')
            prediction_t = remove_batch(scenario.index_time_batched_predicted(output, t))
            scenario.plot_state_rviz(prediction_t, label='predicted', color='blue')

            stepper.step()


def main():
    colorama.init(autoreset=True)

    np.set_printoptions(linewidth=250, precision=4, suppress=True, threshold=10000)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    train_parser.add_argument('checkpoint', type=pathlib.Path)
    train_parser.add_argument('--epochs', type=int, default=500)
    train_parser.set_defaults(func=train_main)

    viz_parser = subparsers.add_parser('viz')
    viz_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    viz_parser.add_argument('checkpoint', type=pathlib.Path)
    viz_parser.set_defaults(func=viz_main)

    args = parser.parse_args()

    from time import time
    now = str(int(time()))
    name = f"train_test_{now}"
    rospy.init_node(name)

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
