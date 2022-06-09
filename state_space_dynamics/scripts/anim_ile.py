#!/usr/bin/env python

import argparse
import pathlib

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
from state_space_dynamics.meta_udnn import compute_batch_time_loss, UDNN
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('checkpoint')

    args = parser.parse_args()

    dataset = TorchDynamicsDataset(fetch_udnn_dataset(args.dataset_dir), mode='notest')
    batch_size = 128
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate, num_workers=get_num_workers(batch_size))

    api = wandb.Api()
    run = api.run(f'armlab/udnn/{args.checkpoint}')
    n_epochs = run.summary['epoch']
    losses = []
    for epoch in trange(n_epochs):
        loss_at_epoch = []
        model_at_epoch = load_model_artifact(args.checkpoint, UDNN, 'udnn', version=f'v{epoch}', user='armlab')
        for inputs in loader:
            outputs = model_at_epoch(inputs)
            batch_time_loss = compute_batch_time_loss(inputs, outputs)
            for time_loss in batch_time_loss:
                for loss in time_loss:
                    loss_at_epoch.append(loss.detach().cpu().numpy())

        losses.append(loss_at_epoch)

    fig = plt.figure()
    ax = plt.gca()
    ax.set_xlabel("training examples, time-ordered")
    ax.set_ylabel("loss")
    ax.set_ylim(bottom=0, top=0.05)

    line, = ax.plot(np.zeros(len(dataset) * dataset.data_collection_params['steps_per_traj']))

    def _viz_epoch(epoch):
        loss_at_epoch = losses[epoch]
        line.set_ydata(loss_at_epoch)
        ax.set_title(f"epoch={epoch}")

    anim = FuncAnimation(fig=fig, func=_viz_epoch, frames=len(losses))
    plt.show()

    print("saving...")
    root = pathlib.Path("results/anim_ile")
    root.mkdir(exist_ok=True, parents=True)
    filename = root / f"anim_ile-{args.dataset_dir}-{args.checkpoint}.gif"
    anim.save(filename.as_posix(), fps=1)


if __name__ == '__main__':
    main()
