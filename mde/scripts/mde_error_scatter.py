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
from link_bot_data.new_dataset_utils import fetch_mde_dataset
from link_bot_pycommon.load_wandb_model import load_model_artifact
from mde.mde_torch import MDE
from mde.torch_mde_dataset import TorchMDEDataset
from moonshine.torch_datasets_utils import my_collate, dataset_shard
from state_space_dynamics.torch_dynamics_dataset import remove_keys


@ros_init.with_ros("mde_error_scatter")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--modes', default='train,val')

    args = parser.parse_args()

    model = load_model_artifact(args.checkpoint, MDE, project='mde', version='best', user='armlab')
    model.eval()

    dataset_dir = fetch_mde_dataset(args.dataset_dir)

    modes = args.modes.split(",")
    for mode in modes:
        transform = transforms.Compose([remove_keys("scene_msg")])
        full_dataset = TorchMDEDataset(dataset_dir, mode=mode, transform=transform)
        s = full_dataset.get_scenario()

        max_len = 1000
        shard = max(int(len(full_dataset) / max_len), 1)
        dataset = dataset_shard(full_dataset, shard)

        pytorch_lightning.seed_everything(1)
        loader = DataLoader(dataset, collate_fn=my_collate, batch_size=16, shuffle=True)

        true_errors = []
        pred_errors = []
        for batch in tqdm(loader):
            true_error = batch['error'][:, 1]
            pred_error = model.forward(batch)
            batch.pop("metadata")

            pred_error = pred_error.detach().cpu().numpy().tolist()
            true_error = true_error.detach().cpu().numpy().tolist()
            for b, (pred_error_i, true_error_i) in enumerate(zip(pred_error, true_error)):
                pred_errors.append(pred_error_i)
                true_errors.append(true_error_i)

                # if true_error_i < 0.05 and pred_error_i > 0.3:
                #     inputs = numpify({k: v[b] for k, v in batch.items()})
                #     time_anim = RvizAnimationController(n_time_steps=2)
                #
                #     time_anim.reset()
                #     while not time_anim.done:
                #         t = time_anim.t()
                #         init_viz_env(s, inputs, t)
                #         full_dataset.transition_viz_t()(s, inputs, t)
                #         s.plot_pred_error_rviz(pred_error_i)
                #         s.plot_error_rviz(true_error_i)
                #         time_anim.step()

        true_errors_2d = np.array(true_errors).reshape([-1, 1])
        pred_errors_2d = np.array(pred_errors).reshape([-1, 1])
        reg = LinearRegression().fit(true_errors_2d, pred_errors_2d)
        r2_score = reg.score(true_errors_2d, pred_errors_2d)
        slope = float(reg.coef_[0, 0])
        print(f"r2_score: {r2_score:.3f}")
        print(f"slope: {slope:.3f}")

        plt.style.use("slides")

        root = pathlib.Path("results/mde_scatters") / args.dataset_dir.name
        root.mkdir(exist_ok=True, parents=True)
        max_error = 0.6

        plt.figure(figsize=(12, 12))
        ax = plt.gca()
        ax.hist(true_errors)
        ax.set_xlim(-0.001, max_error)
        ax.set_title(f"True Error ({mode})")
        ax.set_xlabel("true error")
        ax.set_ylabel("count")

        filename = root / f'true_error_{mode}'
        plt.savefig(filename.as_posix())
        plt.close()

        plt.figure(figsize=(12, 12))
        ax = plt.gca()
        sns.scatterplot(ax=ax, x=true_errors, y=pred_errors, alpha=0.2)
        sns.kdeplot(ax=ax, x=true_errors, y=pred_errors, color='k')
        ax.set_xlim(-0.001, max_error)
        ax.set_ylim(-0.001, max_error)
        ax.set_aspect("equal")
        ax.set_title(f"error ({mode}) ({args.checkpoint})")
        ax.set_xlabel("true error")
        ax.set_ylabel("predicted error")
        ax.text(0.01, 0.9 * max_error, f"r2={r2_score:.3f},slope={slope:.3f}")
        filename = root / f'mde_scatter_{args.checkpoint}_{mode}'
        plt.savefig(filename.as_posix())
        plt.close()


if __name__ == '__main__':
    main()
