#!/usr/bin/env python
from __future__ import print_function

import argparse
from time import time

import matplotlib.pyplot as plt
from colorama import Fore
import numpy as np
import tensorflow as tf

from link_bot_models import constraint_model
from link_bot_models import plotting
from link_bot_models.constraint_model import ConstraintModel, ConstraintModelType
from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset


def plot(args, sdf_data, model, threshold, results, true_positives, true_negatives, false_positives, false_negatives):
    n_examples = results.shape[0]

    sdf_data.image = (sdf_data.image < threshold).astype(np.uint8)

    if args.plot_type == plotting.PlotType.random_individual:
        random_indexes = np.random.choice(n_examples, size=10, replace=False)
        random_results = results[random_indexes]
        figs = []
        for random_result in random_results:
            fig = plotting.plot_single_example(sdf_data, random_result)
            figs.append(fig)
        return plotting.SavableFigureCollection(figs)

    elif args.plot_type == plotting.PlotType.random_combined:
        random_indeces = np.random.choice(n_examples, size=100, replace=False)
        random_results = results[random_indeces]
        savable = plotting.plot_examples(sdf_data.image, sdf_data.extent, random_results, subsample=1, title='random samples')
        return savable

    elif args.plot_type == plotting.PlotType.true_positives:
        savable = plotting.plot_examples(sdf_data.image, sdf_data.extent, true_positives, subsample=5, title='true positives')
        return savable

    elif args.plot_type == plotting.PlotType.true_negatives:
        savable = plotting.plot_examples(sdf_data.image, sdf_data.extent, true_negatives, subsample=5, title='true negatives')
        return savable

    elif args.plot_type == plotting.PlotType.false_positives:
        savable = plotting.plot_examples(sdf_data.image, sdf_data.extent, false_positives, subsample=1, title='false positives')
        return savable

    elif args.plot_type == plotting.PlotType.false_negatives:
        savable = plotting.plot_examples(sdf_data.image, sdf_data.extent, false_negatives, subsample=1, title='false negatives')
        return savable

    elif args.plot_type == plotting.PlotType.interpolate:
        savable = plotting.plot_interpolate(sdf_data, sdf_data.image, model, threshold, title='interpolate')
        return savable

    elif args.plot_type == plotting.PlotType.contours:
        savable = plotting.plot_contours(sdf_data, model, threshold)
        return savable

    elif args.plot_type == plotting.PlotType.animate_contours:
        savable = plotting.animate_contours(sdf_data, model, threshold)
        return savable


def main():
    np.set_printoptions(precision=6, suppress=True)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", type=ConstraintModelType.from_string, choices=list(ConstraintModelType))
    parser.add_argument("dataset", help='use this dataset instead of random rope configurations')
    parser.add_argument("checkpoint", help="eval the *.ckpt name")
    parser.add_argument("threshold", type=float)
    parser.add_argument("plot_type", type=plotting.PlotType.from_string, choices=list(plotting.PlotType))
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--save", action='store_true')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("--debug", help="enable TF Debugger", action='store_true')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random_init", action='store_true')

    args = parser.parse_args()

    # get the rope configurations we're going to evaluate
    dataset = MultiEnvironmentDataset.load_dataset(args.dataset)

    args_dict = vars(args)
    model = ConstraintModel(args_dict, dataset.sdf_shape, args.N)
    model.setup()

    for env_idx, environment in enumerate(dataset.environments):
        print(Fore.GREEN + "Environment {}".format(env_idx) + Fore.RESET)

        sdf_data = environment.sdf_data

        results = constraint_model.evaluate(model, environment)
        m = results.shape[0]

        true_positives = np.array([result for result in results if result.true_violated and result.predicted_violated])
        n_true_positives = len(true_positives)
        false_positives = np.array([result for result in results if result.true_violated and not result.predicted_violated])
        n_false_positives = len(false_positives)
        true_negatives = np.array([result for result in results if not result.true_violated and not result.predicted_violated])
        n_true_negatives = len(true_negatives)
        false_negatives = np.array([result for result in results if not result.true_violated and result.predicted_violated])
        n_false_negatives = len(false_negatives)

        accuracy = (n_true_positives + n_true_negatives) / m
        precision = n_true_positives / (n_true_positives + n_false_positives)
        recall = n_true_negatives / (n_true_negatives + n_false_negatives)
        print("accuracy: {:7.4f}".format(accuracy))
        print("precision: {:7.4f}".format(precision))
        print("recall: {:7.4f}".format(recall))

        savable = plot(args, sdf_data, model, args.threshold, results, true_positives, true_negatives,
                       false_positives, false_negatives)

        plt.show()
        if args.save:
            savable.save('plot_constraint_{}-{}'.format(args.plot_type.name, int(time())))


if __name__ == '__main__':
    main()
