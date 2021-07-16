#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from arc_utilities import ros_init
from learn_invariance.invariance_model import compute_transformation_invariance_error
from link_bot_data.load_dataset import get_dynamics_dataset_loader


@ros_init.with_ros("visualize_invariance_dataset")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory', nargs='+')

    args = parser.parse_args()

    # load the dataset
    dataset_loader = get_dynamics_dataset_loader(args.dataset_dir)
    dataset = dataset_loader.get_datasets(mode='all')
    s = dataset_loader.get_scenario()

    transformations = []
    errors = []
    for inputs in tqdm(dataset.take(1000)):
        transformation = inputs['transformation']
        error = compute_transformation_invariance_error(inputs, s)
        transformations.append(transformation)
        errors.append(error)
    transformations = np.array(transformations)
    errors = np.array(errors)

    angle_sum = np.sum(transformations[:, 3:], axis=-1)
    plt.figure()
    plt.scatter(angle_sum, errors)
    plt.xlabel('roll')
    plt.ylabel('error')

    plt.figure()
    plt.scatter(transformations[:, 4], errors)
    plt.xlabel('pitch')
    plt.ylabel('error')

    plt.figure()
    plt.scatter(transformations[:, 5], errors)
    plt.xlabel('yaw')
    plt.ylabel('error')

    plt.show()


if __name__ == '__main__':
    main()
