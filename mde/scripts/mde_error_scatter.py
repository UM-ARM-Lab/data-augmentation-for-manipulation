#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning
import seaborn as sns
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from arc_utilities import ros_init
from link_bot_pycommon.load_wandb_model import load_model_artifact
from mde.mde_torch import MDE
from mde.torch_mde_dataset import TorchMDEDataset
from moonshine.torch_datasets_utils import my_collate, dataset_shard
from state_space_dynamics.mw_net import MWNet
from state_space_dynamics.torch_dynamics_dataset import remove_keys


@ros_init.with_ros("mde_error_scatter")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('checkpoint', type=str)
    modes = ['train', 'test']

    args = parser.parse_args()

    udnn_meta_checkpoint = "m1_adam_unadapted-470ps"
    udnn_meta_model = load_model_artifact(udnn_meta_checkpoint, MWNet, project='udnn', version='best', user='armlab', train_dataset=None)

    model = load_model_artifact(args.checkpoint, MDE, project='mde', version='best', user='armlab')
    model.eval()

    for mode in modes:
        transform = transforms.Compose([remove_keys("scene_msg")])
        dataset = TorchMDEDataset(args.dataset_dir, mode=mode, transform=transform)

        max_len = 1000
        shard = max(int(len(dataset) / max_len), 1)
        dataset = dataset_shard(dataset, shard)

        pytorch_lightning.seed_everything(1)
        loader = DataLoader(dataset, collate_fn=my_collate, batch_size=16, shuffle=True)

        true_errors = []
        pred_errors = []
        example_indices = []
        for batch in tqdm(loader):
            true_error = batch['error'][:, 1]
            pred_error = model.forward(batch)
            pred_errors.extend(pred_error.detach().cpu().numpy().tolist())
            true_errors.extend(true_error.detach().cpu().numpy().tolist())
            # FIXME: how do we know which MDE example maps to which dynamics dataset example?
            example_indices.extend(batch['example_idx'].detach().cpu().numpy().tolist())

        true_errors_2d = np.array(true_errors).reshape([-1, 1])
        pred_errors_2d = np.array(pred_errors).reshape([-1, 1])
        reg = LinearRegression().fit(true_errors_2d, pred_errors_2d)
        r2_score = reg.score(true_errors_2d, pred_errors_2d)
        slope = float(reg.coef_[0, 0])
        print(f"r2_score: {r2_score:.3f}")
        print(f"slope: {slope:.3f}")

        learned_data_weights = udnn_meta_model.sample_weights[example_indices]

        plt.style.use("slides")
        plt.figure(figsize=(12, 12))
        ax = plt.gca()
        sns.scatterplot(ax=ax, x=true_errors, y=pred_errors, hue=learned_data_weights, alpha=0.1)
        sns.kdeplot(ax=ax, x=true_errors, y=pred_errors, color='k')
        y_max = 0.15
        ax.set_xlim(-0.001, 0.15)
        ax.set_ylim(-0.001, y_max)
        ax.set_aspect("equal")
        ax.set_title(f"Error ({mode}) ({args.checkpoint})")
        ax.set_xlabel("true error")
        ax.set_ylabel("predicted error")
        ax.text(0.01, 0.9 * y_max, f"r2={r2_score:.3f},slope={slope:.3f}")

        root = pathlib.Path("results/mde_scatters")
        root.mkdir(exist_ok=True, parents=True)
        filename = root / f'mde_scatter_{args.checkpoint}_{mode}'
        plt.savefig(filename.as_posix())
        plt.close()


if __name__ == '__main__':
    main()
