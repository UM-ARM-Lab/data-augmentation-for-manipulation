#!/usr/bin/env python
import argparse
import pathlib

import colorama
import numpy as np
from colorama import Style, Fore

from arc_utilities import ros_init
from link_bot_planning.analysis import results_utils
from link_bot_planning.analysis.results_utils import classifier_params_from_planner_params, plot_steps
from link_bot_planning.plan_and_execute import TrialStatus
from link_bot_pycommon.args import my_formatter


@ros_init.with_ros("plot_ift_results")
def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("ift_dir", type=pathlib.Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--full-plan", action='store_true')
    parser.add_argument("--only-timeouts", action='store_true')
    parser.add_argument("--verbose", '-v', action="count", default=0)

    args = parser.parse_args()

    planning_results_dir = args.ift_dir / 'planning_results'
    for results_dir in sorted(planning_results_dir.iterdir()):
        if not results_dir.is_dir():
            continue

        print(Style.BRIGHT + Fore.GREEN + results_dir.name + Style.RESET_ALL)
        scenario, metadata = results_utils.get_scenario_and_metadata(results_dir)
        classifier_params = classifier_params_from_planner_params(metadata['planner_params'])
        if args.threshold is None:
            threshold = classifier_params['classifier_dataset_hparams']['labeling_params']['threshold']
        else:
            threshold = args.threshold

        for trial_idx, datum in results_utils.trials_generator(results_dir):
            should_skip = args.only_timeouts and datum['trial_status'] != TrialStatus.Timeout
            if should_skip:
                continue

            msg = f"trial {trial_idx}, status {datum['trial_status']}"
            print(Fore.LIGHTBLUE_EX + msg)
            plot_steps(scenario, datum, metadata, {'threshold': threshold}, args.verbose, args.full_plan)


if __name__ == '__main__':
    main()
