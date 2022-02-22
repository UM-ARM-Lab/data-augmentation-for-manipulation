import argparse
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
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

        data[ckpt] = (np.array(weights), np.array(losses))

    data_sorted = {}
    for ckpt, (weights, losses) in data.items():
        sorting_criteria = 1000 * weights - losses
        indices_sorted = np.argsort(sorting_criteria)
        data_sorted[ckpt] = (weights[indices_sorted], losses[indices_sorted])

    plt.style.use('paper')
    plt.rcParams['figure.figsize'] = (15, 8)
    color_map = cm.RdYlGn

    plt.figure()
    for ckpt, (weights, losses) in data_sorted.items():
        plt.plot(losses, label=ckpt)

    ymin, ymax = plt.ylim()
    weights_img = np.expand_dims(weights, 0)
    plt.imshow(weights_img, cmap=color_map, extent=[-0.5, len(dataset)-0.5, ymin, ymax], alpha=0.5)

    plt.colorbar()
    plt.title("Dynamics Prediction Accuracy")
    plt.ylabel("loss")
    plt.xlabel("test example")
    plt.yscale('log')
    plt.legend()
    plt.savefig("dynamics_prediction_accuracy.png")
    plt.show()


if __name__ == '__main__':
    main()
