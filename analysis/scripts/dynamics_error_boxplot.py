#!/usr/bin/env python

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from analysis.analyze_results import try_split_model_name
from link_bot_data.new_dataset_utils import fetch_udnn_dataset
from moonshine.moonshine_utils import get_num_workers
from moonshine.torch_datasets_utils import my_collate
from state_space_dynamics.torch_dynamics_dataset import remove_keys, TorchDynamicsDataset
from state_space_dynamics.train_test_dynamics import load_udnn_model_wrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('checkpoints', type=str, nargs='+')
    parser.add_argument('--mode', type=str, default='train')

    args = parser.parse_args()

    transform = transforms.Compose([remove_keys("scene_msg")])
    dataset_dir = fetch_udnn_dataset(args.dataset_dir)
    dataset = TorchDynamicsDataset(dataset_dir, mode=args.mode, transform=transform)
    batch_size = 8
    loader = DataLoader(dataset, collate_fn=my_collate, batch_size=batch_size, num_workers=get_num_workers(batch_size))

    data = []
    for checkpoint in args.checkpoints:
        checkpoint, legend_name = try_split_model_name(checkpoint)
        model = load_udnn_model_wrapper(checkpoint)
        model.eval()
        for example in tqdm(loader):
            outputs = model(example)
            error_batch = model.scenario.classifier_distance_torch(example, outputs)
            for error_time in error_batch.detach().numpy().squeeze().tolist():
                for error_t in error_time:
                    data.append([checkpoint, legend_name, error_t])

    df = pd.DataFrame(data, columns=['checkpoint', 'legend_name', 'dynamics_error'])

    plt.style.use("slides")
    # sns.boxplot(data=df, y='dynamics_error', x='legend_name')
    sns.boxenplot(data=df, y='dynamics_error', x='legend_name', k_depth='full')
    plt.savefig("results/dynamics_error.png")
    plt.show(block=True)


if __name__ == '__main__':
    main()
