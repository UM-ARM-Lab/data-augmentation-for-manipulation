import argparse
from matplotlib import cm
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from link_bot_pycommon.load_wandb_model import load_model_artifact
from moonshine.moonshine_utils import get_num_workers
from moonshine.torch_datasets_utils import my_collate
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset, remove_keys
from state_space_dynamics.udnn_torch import UDNN


def main():
    np.set_printoptions(linewidth=300)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('checkpoints', nargs='+')
    parser.add_argument('--mode', default='test')
    parser.add_argument('--batch-size', type=int, default=16)

    args = parser.parse_args()

    transform = transforms.Compose([remove_keys("scene_msg")])
    dataset = TorchDynamicsDataset(args.dataset_dir, args.mode, transform=transform)
    loader = DataLoader(dataset, collate_fn=my_collate, num_workers=get_num_workers(args.batch_size))

    data = {}
    for ckpt in args.checkpoints:
        model: UDNN = load_model_artifact(ckpt, UDNN, project='udnn', version='best', user='armlab')

        losses = []
        weights = []
        for inputs in loader:
            outputs = model(inputs)
            loss = model.compute_loss(inputs, outputs).detach().cpu().numpy().squeeze()
            losses.append(loss)
            weights.append(float(inputs['weight'].numpy()))

        data[ckpt] = (weights, losses)

    plt.style.use('paper')
    plt.rcParams['figure.figsize'] = (8, 5)
    color_map = cm.RdYlGn

    plt.figure()
    for ckpt, (weights, losses) in data.items():
        indices = np.arange(len(weights))
        plt.plot(losses, label=ckpt)
        plt.scatter(indices, losses, c=weights, cmap=color_map)

    plt.colorbar()
    plt.title("Dynamics Prediction Accuracy")
    plt.ylabel("loss")
    plt.xlabel("test example")
    plt.yscale('log')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
