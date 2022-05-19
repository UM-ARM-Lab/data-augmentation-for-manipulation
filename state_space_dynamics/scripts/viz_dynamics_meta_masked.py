#!/usr/bin/env python
import argparse
import pathlib

import numpy as np

from arc_utilities import ros_init
from link_bot_data.visualization import init_viz_env, viz_pred_actual_t
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.numpify import numpify
from moonshine.torch_and_tf_utils import remove_batch, add_batch
from moonshine.torchify import torchify
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset
from state_space_dynamics.train_test_dynamics import load_udnn_model_wrapper


def viz_main(dataset_dir: pathlib.Path, checkpoint, mode: str):
    dataset = TorchDynamicsDataset(dataset_dir, mode)

    model = load_udnn_model_wrapper(checkpoint)

    s = dataset.get_scenario()

    dataset_anim = RvizAnimationController(n_time_steps=len(dataset), ns='trajs')

    while not dataset_anim.done:
        inputs = dataset[dataset_anim.t()]

        outputs = remove_batch(model(torchify(add_batch(inputs))))
        error = model.scenario.classifier_distance(inputs, numpify(outputs))
        time_steps_to_show = np.argwhere(np.logical_or(inputs['meta_mask'][1:], inputs['meta_mask'][:-1]))
        time_anim = RvizAnimationController(time_steps=time_steps_to_show)

        while not time_anim.done:
            t = time_anim.t()
            init_viz_env(s, inputs, t)
            viz_pred_actual_t(dataset, model, inputs, outputs, s, t, threshold=0.05)
            time_anim.step()

    dataset_anim.step()


@ros_init.with_ros('viz_dynamics_meta_masked')
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--mode', type=str, default='test')

    args = parser.parse_args()

    viz_main(args.dataset_dir, args.checkpoint, mode=args.mode)


if __name__ == '__main__':
    main()
