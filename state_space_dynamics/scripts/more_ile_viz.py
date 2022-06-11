#!/usr/bin/env python

import argparse
import pathlib

import numpy as np
import wandb

from arc_utilities import ros_init
from link_bot_data.new_dataset_utils import fetch_udnn_dataset
from link_bot_data.visualization import init_viz_env, viz_pred_actual_t
from link_bot_pycommon.load_wandb_model import load_model_artifact
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.numpify import numpify
from moonshine.torch_and_tf_utils import add_batch, remove_batch
from moonshine.torchify import torchify
from state_space_dynamics.meta_udnn import UDNN
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset


@ros_init.with_ros("more_ile_viz")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('checkpoint')

    args = parser.parse_args()

    dataset = TorchDynamicsDataset(fetch_udnn_dataset(args.dataset_dir), mode='notest')
    s = dataset.get_scenario()
    example_idx = 0
    inputs = dataset[example_idx]

    root = pathlib.Path("results/anim_ile")
    root.mkdir(exist_ok=True, parents=True)

    api = wandb.Api()
    run = api.run(f'armlab/udnn/{args.checkpoint}')
    n_epochs = run.summary['epoch']

    initial_model, _, initial_errors = get_data_for_epoch(args.checkpoint, 0, inputs, s)
    time_steps = np.argwhere(initial_errors < .08).squeeze(1)
    viz_inputs(dataset, inputs, initial_model, time_steps)

    final_model, _, final_errors = get_data_for_epoch(args.checkpoint, n_epochs - 1, inputs, s)
    time_steps = np.argwhere(final_errors < .08).squeeze()
    viz_inputs(dataset, inputs, final_model, time_steps)


def get_data_for_epoch(checkpoint, epoch, inputs, s):
    model_at_epoch = load_model_artifact(checkpoint, UDNN, 'udnn', version=f'v{epoch}', user='armlab')

    outputs = numpify(remove_batch(model_at_epoch(torchify(add_batch(inputs)))))
    error_at_epoch = s.classifier_distance(inputs, outputs)[1:]  # skip t=0
    xs = np.arange(error_at_epoch.shape[0])

    return model_at_epoch, xs, error_at_epoch


def viz_inputs(dataset, inputs, model, time_steps):
    s = model.scenario

    time_anim = RvizAnimationController(time_steps=time_steps)
    outputs = remove_batch(model(torchify(add_batch(inputs))))

    time_anim.reset()
    while not time_anim.done:
        t = time_anim.t()
        init_viz_env(s, inputs, t)
        viz_pred_actual_t(dataset, model, inputs, outputs, s, t, threshold=0.08)
        time_anim.step()


#
#         with saved_results.open("wb") as f:
#             pickle.dump([errors, initial_errors_sorted], f)
#     return dataset, xs, errors, initial_errors_sorted, root
#
#
# def make_animation(args, xs, errors, initial_errors_sorted, root):
#     fig = plt.figure(figsize=(10, 4))
#     ax = plt.gca()
#     ax.set_xlabel("training data, sorted by initial error")
#     ax.set_ylabel("error")
#     ax.set_ylim(bottom=9e-3, top=1.3)
#     ax.set_yscale("log")
#     line, = ax.plot(xs, np.ones(len(errors[0])), label='current')
#     ax.plot(xs, initial_errors_sorted, label='initial error')
#     ax.legend()
#     ax.axhline(y=0.08, color='k', linestyle='--')
#
#     def _viz_epoch(frame_idx):
#         error_at_epoch = errors[frame_idx]
#         line.set_ydata(error_at_epoch)
#         ax.set_title(f"{frame_idx}")
#
#     anim = FuncAnimation(fig=fig, func=_viz_epoch, frames=len(errors))
#     filename = root / f"anim_ile-{args.dataset_dir}-{args.checkpoint}-{int(time())}.gif"
#     print(f"saving {filename.as_posix()}")
#     anim.save(filename.as_posix(), fps=1)
#     plt.close()
#     print("done")


if __name__ == '__main__':
    main()
