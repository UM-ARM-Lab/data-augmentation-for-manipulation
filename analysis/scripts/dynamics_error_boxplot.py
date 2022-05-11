#!/usr/bin/env python

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from moonshine.moonshine_utils import get_num_workers
from moonshine.torch_datasets_utils import my_collate
from state_space_dynamics.torch_dynamics_dataset import remove_keys, TorchDynamicsDataset
from state_space_dynamics.train_test_dynamics import load_udnn_model_wrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('checkpoints', type=str, nargs='+')
    parser.add_argument('--mode', type=str, default='all')

    args = parser.parse_args()

    transform = transforms.Compose([remove_keys("scene_msg")])
    dataset = TorchDynamicsDataset(args.dataset_dir, mode=args.mode, transform=transform)
    loader = DataLoader(dataset, collate_fn=my_collate, num_workers=get_num_workers(32))

    data = []
    for checkpoint in args.checkpoints:
        model = load_udnn_model_wrapper(checkpoint)
        model.eval()
        for example in tqdm(loader):
            outputs = model(example)
            error_batch = model.scenario.classifier_distance_torch(example, outputs)
            for error in error_batch.detach().numpy().squeeze().tolist():
                data.append([checkpoint, error])

    df = pd.DataFrame(data, columns=['checkpoint', 'dynamics_error'])

    plt.style.use("slides")
    sns.boxplot(data=df, y='dynamics_error', x='checkpoint')
    plt.show()


if __name__ == '__main__':
    main()
