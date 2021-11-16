#!/usr/bin/env python
import argparse
import pathlib

import numpy as np

import colorama

import rospy
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.moonshine_utils import remove_batch
from moonshine.numpify import numpify
from state_space_dynamics.train_test_dynamics import viz_dataset


def viz_func(batch, predictions, test_dataset: DynamicsDatasetLoader):
    """ we assume batch size of 1 """
    test_dataset.scenario.plot_environment_rviz(remove_batch(batch))
    anim = RvizAnimationController(np.arange(test_dataset.steps_per_traj))
    while not anim.done:
        t = anim.t()
        actual_t = numpify(remove_batch(test_dataset.scenario.index_time_batched_predicted(batch, t)))
        action_t = numpify(remove_batch(test_dataset.scenario.index_time_batched_predicted(batch, t)))
        test_dataset.scenario.plot_state_rviz(actual_t, label='actual', color='red')
        test_dataset.scenario.plot_action_rviz(actual_t, action_t, color='gray')
        prediction_t = remove_batch(test_dataset.scenario.index_time_batched_predicted(predictions, t))
        test_dataset.scenario.plot_state_rviz(prediction_t, label='predicted', color='blue')

        anim.step()


def main():
    colorama.init(autoreset=True)

    np.set_printoptions(linewidth=250, precision=4, suppress=True, threshold=10000)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('checkpoint', type=pathlib.Path)
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'val'], default='test')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    rospy.init_node('viz_obs')

    viz_dataset(dataset_dirs=args.dataset_dirs,
                checkpoint=args.checkpoint,
                mode=args.mode,
                viz_func=viz_func)


if __name__ == '__main__':
    main()
