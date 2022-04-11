#!/usr/bin/env python
import argparse
import pathlib

from arc_utilities import ros_init
from link_bot_data.visualization import init_viz_env, viz_pred_t, viz_actual_t
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.torch_and_tf_utils import remove_batch, add_batch
from moonshine.torchify import torchify
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset
from state_space_dynamics.train_test_dynamics import load_udnn_model_wrapper


def viz_compare_models(dataset_dir, checkpoints, mode):
    dataset = TorchDynamicsDataset(dataset_dir, mode)

    color_map = {
        'm1_notest-5t0bf':         '#E57439',
        'm1_adam_unadapted-470ps': '#A0C75C',
        'robot-8qa3q':             '#EDB732',
    }
    models = []
    colors = []
    for checkpoint in checkpoints:
        colors.append(color_map.get(checkpoint, 'black'))
        model = load_udnn_model_wrapper(checkpoint)
        model.eval()
        models.append(model)

    s = dataset.get_scenario()

    dataset_anim = RvizAnimationController(n_time_steps=len(dataset), ns='trajs')
    time_anim = RvizAnimationController(n_time_steps=10)

    n_examples_visualized = 0
    while not dataset_anim.done:
        inputs = dataset[dataset_anim.t()]

        outputs_list = [remove_batch(model(torchify(add_batch(inputs)))) for model in models]

        time_anim.reset()
        while not time_anim.done:
            t = time_anim.t()
            init_viz_env(s, inputs, t)
            for outputs, model, checkpoint, color in zip(outputs_list, models, checkpoints, colors):
                viz_actual_t(dataset, inputs, s, t)
                viz_pred_t(dataset, model, inputs, outputs, s, t, threshold=0.05, color=color, label=checkpoint)
            time_anim.step()

        n_examples_visualized += 1

        dataset_anim.step()

    print(f"{n_examples_visualized:=}")


@ros_init.with_ros('viz_compare_models')
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('checkpoints', type=str, nargs='+')
    parser.add_argument('--mode', type=str, default='test')

    args = parser.parse_args()

    viz_compare_models(args.dataset_dir, args.checkpoints, args.mode)


if __name__ == '__main__':
    main()
