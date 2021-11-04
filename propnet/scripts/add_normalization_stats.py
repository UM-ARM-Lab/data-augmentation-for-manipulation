#!/usr/bin/env python
import argparse
import pathlib
import pickle

import torch
from tqdm import tqdm

from arc_utilities import ros_init
from propnet.torch_dynamics_dataset import TorchDynamicsDataset, remove_keys


@ros_init.with_ros("modify_dynamics_dataset")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')

    args = parser.parse_args()

    dataset = TorchDynamicsDataset(args.dataset_dir,
                                   mode='train',
                                   transform=remove_keys('filename', 'full_filename', 'joint_names', 'metadata'),
                                   add_stats=False)  # don't load the existing stats
    n = len(dataset)
    features_sums = {}

    for e in tqdm(dataset):
        for k, v in e.items():
            if k not in features_sums:
                features_sums[k] = {'sum':         torch.zeros(v.shape),
                                    'squared_sum': torch.zeros(v.shape)}

            features_sums[k]['sum'] += v
            features_sums[k]['squared_sum'] += v ** 2

    feature_stats = {}
    for k, feature_sums in features_sums.items():
        feature_sum = feature_sums['sum']
        feature_squared_sum = feature_sums['squared_sum']
        mean = feature_sum / n
        variance = (feature_squared_sum / n - (mean ** 2))
        # some feature have near zero variance, but we can't sqrt a negative
        variance += 1e-6
        std = variance ** 0.5
        feature_stats[k] = (mean, std, n)

    stats_filename = args.dataset_dir / 'stats.pkl'
    with stats_filename.open("wb") as f:
        pickle.dump(feature_stats, f)


if __name__ == '__main__':
    main()
