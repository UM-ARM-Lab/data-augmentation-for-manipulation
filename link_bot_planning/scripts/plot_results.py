#!/usr/bin/env python
import argparse
import pathlib

import colorama
import numpy as np

from arc_utilities import ros_init
from link_bot_planning.analysis import results_utils
from link_bot_planning.analysis.results_utils import classifier_params_from_planner_params, plot_steps
from link_bot_planning.plan_and_execute import TrialStatus
from link_bot_pycommon.args import my_formatter, int_set_arg


@ros_init.with_ros("plot_results")
def main():
    colorama.init(autoreset=True)
    np.set_printoptions(linewidth=250, precision=3, suppress=True)
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("results_dir", type=pathlib.Path, help='directory containing metrics.json')
    parser.add_argument("--trial-indices", type=int_set_arg, help='which plan(s) to show')
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--full-plan", action='store_true')
    parser.add_argument("--only-timeouts", action='store_true')
    parser.add_argument("--verbose", '-v', action="count", default=0)

    args = parser.parse_args()

    scenario, metadata = results_utils.get_scenario_and_metadata(args.results_dir)
    classifier_params = classifier_params_from_planner_params(metadata['planner_params'])
    if args.threshold is None:
        threshold = classifier_params['classifier_dataset_hparams']['labeling_params']['threshold']
    else:
        threshold = args.threshold

    for trial_idx, datum in results_utils.trials_generator(args.results_dir, args.trial_indices):
        should_skip = args.only_timeouts and datum['trial_status'] != TrialStatus.Timeout
        if should_skip:
            continue

        print(f"trial {trial_idx} ...")
        plot_steps(scenario, datum, metadata, {'threshold': threshold}, args.verbose, args.full_plan)
        print(f"... complete with status {datum['trial_status']}")


if __name__ == '__main__':
    main()
