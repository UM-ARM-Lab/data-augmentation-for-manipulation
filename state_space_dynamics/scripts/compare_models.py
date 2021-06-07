#!/usr/bin/env python
import argparse
import gzip
import json
import pathlib
import time

import colorama
import numpy as np
from colorama import Fore, Style
from matplotlib import cm
from tabulate import tabulate

import rospy
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_data.dataset_utils import batch_tf_dataset
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.metric_utils import row_stats, dict_to_pvalue_table
from link_bot_pycommon.pycommon import paths_from_json
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import listify, numpify, remove_batch
from state_space_dynamics import dynamics_utils

limit_gpu_mem(8.5)


def load_dataset_and_models(args):
    comparison_info = json.load(args.comparison.open("r"))
    models = {}
    for name, model_info in comparison_info.items():
        model_dir = paths_from_json(model_info['model_dir'])
        model = dynamics_utils.load_generic_model(model_dir)
        models[name] = model

    dataset = DynamicsDatasetLoader(args.dataset_dirs)
    tf_dataset = dataset.get_datasets(mode=args.mode,
                                      shard=args.shard,
                                      take=args.take)
    tf_dataset = batch_tf_dataset(tf_dataset, 1)

    return tf_dataset, dataset, models


def generate(args):
    base_folder = pathlib.Path('results') / 'compare_models' / \
                  '{}-{}-{}'.format(args.nickname, args.mode, int(time.time()))
    base_folder.mkdir(parents=True)
    print("Using output directory: {}".format(base_folder))

    tf_dataset, dataset, models = load_dataset_and_models(args)

    all_data = []
    for example_idx, dataset_element in enumerate(tf_dataset):
        dataset_element.update(dataset.batch_metadata)
        data_per_model_for_element = {
            'time_steps': dataset.steps_per_traj,
            'action_keys': dataset.action_keys,
            'dataset_element': dataset_element,
            'environment': {
                'env': dataset_element['env'],
                'res': dataset_element['res'],
                'origin': dataset_element['origin'],
                'extent': dataset_element['extent'],
            }
        }
        print(f'running prediction for example {example_idx}')

        for model_name, model in models.items():
            predictions = model.propagate_from_example(dataset_element)
            data_per_model_for_element[model_name] = {
                'predictions': predictions,
                'scenario': model.scenario.simple_name(),
            }
        all_data.append(data_per_model_for_element)
    results_filename = base_folder / 'saved_data.json.gz'
    print(Fore.GREEN + "Saving results to {}".format(results_filename) + Fore.RESET)
    with gzip.open(results_filename, "wb") as results_file:
        results_str = json.dumps(listify(all_data))
        results_file.write(results_str.encode("utf-8"))
    viz(results_filename, args.fps, args.no_plot, args.save)


def viz_main(args):
    viz(args.data_filename, args.fps, args.no_plot, args.save)


def viz(data_filename, fps, no_plot, save):
    rospy.init_node("compare_models")

    # Load the results
    base_folder = data_filename.parent
    with gzip.open(data_filename, "rb") as data_file:
        data_str = data_file.read()
        saved_data = json.loads(data_str.decode("utf-8"))

    all_metrics = {}
    for example_idx, datum in enumerate(saved_data):
        print(example_idx)

        # use the first (or any) model data to get the ground truth and
        dataset_element = numpify(datum.pop("dataset_element"))
        environment = numpify(datum.pop("environment"))
        action_keys = datum.pop("action_keys")
        actions = {k: dataset_element[k] for k in action_keys}

        models_viz_info = {}
        n_models = len(datum)
        time_steps = np.arange(datum.pop('time_steps'))
        for model_name, data_for_model in datum.items():
            scenario = get_scenario(data_for_model['scenario'])

            # Metrics
            metrics_for_model = {}
            predictions = numpify(data_for_model['predictions'])
            predictions.pop('stdev')
            metrics = scenario.dynamics_metrics_function(dataset_element, predictions)
            loss = scenario.dynamics_loss_function(dataset_element, predictions)
            metrics['loss'] = loss
            for metric_name, metric_value in metrics.items():
                if metric_name not in metrics_for_model:
                    metrics_for_model[metric_name] = []
                metrics_for_model[metric_name].append(metric_value.numpy())

            for metric_name, metric_values in metrics_for_model.items():
                mean_metric_value = float(np.mean(metric_values))
                if model_name not in all_metrics:
                    all_metrics[model_name] = {}
                if metric_name not in all_metrics[model_name]:
                    all_metrics[model_name][metric_name] = []
                all_metrics[model_name][metric_name].append(mean_metric_value)

            models_viz_info[model_name] = (scenario, predictions)

        if not no_plot and not save:
            # just use whatever the latest scenario was, it shouldn't matter which we use
            scenario.plot_environment_rviz(remove_batch(environment))
            anim = RvizAnimationController(time_steps)
            while not anim.done:
                t = anim.t()
                actual_t = remove_batch(scenario.index_state_time(dataset_element, t))
                action_t = remove_batch(scenario.index_action_time(actions, t))
                scenario.plot_state_rviz(actual_t, label='actual', color='#0000ff88')
                scenario.plot_action_rviz(actual_t, action_t, color='gray')
                for model_idx, (model_name, viz_info) in enumerate(models_viz_info.items()):
                    scenario_i, predictions = viz_info
                    prediction_t = remove_batch(scenario_i.index_state_time(predictions, t))
                    color = cm.jet(model_idx / n_models)
                    scenario_i.plot_state_rviz(prediction_t, label=model_name, color=color)

                anim.step()

    metrics_by_model = {}
    for model_name, metrics_for_model in all_metrics.items():
        for metric_name, metric_values in metrics_for_model.items():
            if metric_name not in metrics_by_model:
                metrics_by_model[metric_name] = {}
            metrics_by_model[metric_name][model_name] = metric_values

    with (base_folder / 'metrics_tables.txt').open("w") as metrics_file:
        for metric_name, metric_by_model in metrics_by_model.items():
            headers = ["Model", "min", "max", "mean", "median", "std"]
            table_data = []
            for model_name, metric_values in metric_by_model.items():
                table_data.append([model_name] + row_stats(metric_values))
            print('-' * 90)
            print(Style.BRIGHT + metric_name + Style.NORMAL)
            table = tabulate(table_data,
                             headers=headers,
                             tablefmt='fancy_grid',
                             floatfmt='6.4f',
                             numalign='center',
                             stralign='left')
            metrics_file.write(table)
            print(table)
            print()

            print(Style.BRIGHT + f"p-value matrix [{metric_name}]" + Style.NORMAL)
            print(dict_to_pvalue_table(metric_by_model))


def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    subparsers = parser.add_subparsers()
    gen_parser = subparsers.add_parser('generate')
    gen_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+', help='dataset dirs')
    gen_parser.add_argument('comparison', type=pathlib.Path, help='json file describing what should be compared')
    gen_parser.add_argument('nickname', help='used to name output directory')
    gen_parser.add_argument('--sequence-length', type=int, default=10, help='seq length')
    gen_parser.add_argument('--no-plot', action='store_true', help='no plot')
    gen_parser.add_argument('--mode', choices=['train', 'test', 'val'], default='test', help='mode')
    gen_parser.add_argument('--take', type=int, default=100, help='number of examples to visualize')
    gen_parser.add_argument('--shard', type=int, help='shard only a subset of the data')
    gen_parser.add_argument('--shuffle', action='store_true')
    gen_parser.add_argument('--save', action='store_true')
    gen_parser.add_argument('--fps', type=float, default=1)
    gen_parser.set_defaults(func=generate)

    viz_parser = subparsers.add_parser('viz')
    viz_parser.add_argument('data_filename', type=pathlib.Path, help='saved data from generate')
    viz_parser.add_argument('--fps', type=float, default=1)
    viz_parser.add_argument('--save', action='store_true')
    viz_parser.add_argument('--no-plot', action='store_true')
    viz_parser.set_defaults(func=viz_main)

    args = parser.parse_args()
    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
