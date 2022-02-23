import argparse
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from analysis.results_figures import boxplot
from link_bot_pycommon.load_wandb_model import load_model_artifact
from link_bot_pycommon.pandas_utils import df_where
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

    data = []
    for ckpt in args.checkpoints:
        model: UDNN = load_model_artifact(ckpt, UDNN, project='udnn', version='best', user='armlab')

        for inputs in tqdm(loader):
            outputs = model(inputs)
            loss = model.compute_batch_time_loss(inputs, outputs).detach().cpu().numpy().squeeze()
            # loss is [time] shaped, so is weight, and we want to skip t=0 where loss is always 0
            for loss_t, weight_t in list(zip(loss, inputs['weight'].numpy().squeeze()))[1:]:
                data.append([ckpt, weight_t, loss_t])

    df = pd.DataFrame(data, columns=['ckpt', 'weight', 'loss'])

    df_weight_1 = df_where(df, 'weight', 1.0)
    df_weight_0 = df_where(df, 'weight', 0.0)

    plt.style.use('paper')
    plt.rcParams['figure.figsize'] = (15, 8)

    fig, ax = boxplot(df_weight_0, pathlib.Path("."), 'ckpt', 'loss', 'weight = 0', save=False)
    fig, ax = boxplot(df_weight_1, pathlib.Path("."), 'ckpt', 'loss', 'weight = 1', save=False)

    plt.show()


if __name__ == '__main__':
    main()
