#!/usr/bin/env python

import argparse
import pathlib
import pickle
from time import time

import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.animation import FuncAnimation
from torch.utils.data import DataLoader
from tqdm import trange

from link_bot_data.new_dataset_utils import fetch_udnn_dataset
from link_bot_pycommon.load_wandb_model import load_model_artifact
from moonshine.moonshine_utils import get_num_workers
from moonshine.torch_datasets_utils import my_collate
from state_space_dynamics.meta_udnn import UDNN
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('checkpoint')
    parser.add_argument('--regenerate', action='store_true')

    args = parser.parse_args()

    dataset, errors, initial_errors_sorted, root = compute_data(args)

    make_animation(args, errors, initial_errors_sorted, root)


def compute_data(args):
    dataset = TorchDynamicsDataset(fetch_udnn_dataset(args.dataset_dir), mode='notest')
    s = dataset.get_scenario()
    batch_size = 12
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate, num_workers=get_num_workers(batch_size))
    root = pathlib.Path("results/anim_ile")
    root.mkdir(exist_ok=True, parents=True)
    saved_results = root / 'losses.pkl'
    if saved_results.exists() and not args.regenerate:
        with saved_results.open("rb") as f:
            errors, initial_errors_sorted = pickle.load(f)
    else:
        api = wandb.Api()
        run = api.run(f'armlab/udnn/{args.checkpoint}')
        n_epochs = run.summary['epoch']
        errors = []
        initial_errors_sorted = None
        sorted_indices = None
        start_epoch = 100
        for epoch in trange(start_epoch, n_epochs, 4):
            model_at_epoch = load_model_artifact(args.checkpoint, UDNN, 'udnn', version=f'v{epoch}', user='armlab')
            error_at_epoch = []
            for inputs in loader:
                outputs = model_at_epoch(inputs)
                batch_time_error = s.classifier_distance_torch(inputs, outputs)[:, 1:]  # skip t=0
                error_at_epoch.extend(
                    batch_time_error.flatten().detach().cpu().numpy().tolist()
                )
                # batch_error = batch_time_error.mean(-1)  # average over time
                # for error in batch_error:
                #     error_at_epoch.append(float(error.detach().cpu().numpy()))
                # for time_error in batch_time_error:
                #     for error in time_error:
                #         error_at_epoch.append(float(error.detach().cpu().numpy()))

            error_at_epoch = np.array(error_at_epoch)
            if epoch == start_epoch:
                sorted_indices = np.argsort(error_at_epoch)
                error_at_epoch_sorted = error_at_epoch[sorted_indices]
                initial_errors_sorted = error_at_epoch_sorted
            else:
                error_at_epoch_sorted = error_at_epoch[sorted_indices]

            errors.append(error_at_epoch_sorted)

        with saved_results.open("wb") as f:
            pickle.dump([errors, initial_errors_sorted], f)
    return dataset, errors, initial_errors_sorted, root


def make_animation(args, errors, initial_errors_sorted, root):
    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.set_xlabel("training trajectories, sorted by initial error")
    ax.set_ylabel("error")
    ax.set_xticklabels([])
    ax.set_ylim(bottom=1e-2, top=1.2)
    ax.set_yscale("log")
    line, = ax.plot(np.ones(len(errors[0])), label='current')
    ax.plot(initial_errors_sorted, label='initial error')
    ax.legend()
    ax.axhline(y=0.08, color='k', linestyle='--')

    def _viz_epoch(frame_idx):
        error_at_epoch = errors[frame_idx]
        line.set_ydata(error_at_epoch)
        ax.set_title(f"{frame_idx}")

    anim = FuncAnimation(fig=fig, func=_viz_epoch, frames=len(errors))
    filename = root / f"anim_ile-{args.dataset_dir}-{args.checkpoint}-{int(time())}.gif"
    print(f"saving {filename.as_posix()}")
    anim.save(filename.as_posix(), fps=24)
    plt.close()
    print("done")


if __name__ == '__main__':
    main()
