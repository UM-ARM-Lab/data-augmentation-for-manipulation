import argparse
import pathlib

import matplotlib.pyplot as plt
import pytorch_lightning
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from arc_utilities import ros_init
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
    parser.add_argument('--mode', type=str, default='test')

    args = parser.parse_args()

    transform = transforms.Compose([remove_keys("scene_msg")])
    dataset = TorchMDEDataset(args.dataset_dir, mode=args.mode, transform=transform)

    model = load_model_artifact(args.checkpoint, MDE, project='mde', version='best', user='armlab')
    model.eval()

    max_len = 1000
    shard = max(int(len(dataset) / max_len), 1)
    dataset = dataset_shard(dataset, shard)

    pytorch_lightning.seed_everything(1)
    loader = DataLoader(dataset, collate_fn=my_collate, batch_size=16, shuffle=True)

    true_errors = []
    pred_errors = []
    for batch in tqdm(loader):
        true_error = batch['error'][:, 1]
        pred_error = model.forward(batch)
        pred_errors.extend(pred_error.detach().cpu().numpy().tolist())
        true_errors.extend(true_error.detach().cpu().numpy().tolist())

    plt.figure()
    plt.axis('equal')
    ax = plt.gca()
    ax.scatter(true_errors, pred_errors, c='k', alpha=0.1)
    sns.kdeplot(ax=ax, x=true_errors, y=pred_errors)
    ax.set_xlim(left=-0.001)
    ax.set_ylim(bottom=-0.001)
    ax.set_title("True vs Predicted Model Error")
    ax.set_xlabel("true error")
    ax.set_ylabel("predicted error")

    plt.pause(2)
    plt.show(block=True)


if __name__ == '__main__':
    main()
