#!/usr/bin/env python
import argparse
import pathlib

import colorama
from colorama import Style, Fore

from arc_utilities import ros_init
from analysis import results_utils
from analysis.results_utils import classifier_params_from_planner_params, plot_steps
from link_bot_planning.trial_result import TrialStatus
from link_bot_pycommon.screen_recorder import ScreenRecorder
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)

@ros_init.with_ros("plot_ift_results")
def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("ift_dir", type=pathlib.Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--full-plan", action='store_true')
    parser.add_argument("--record", action='store_true')
    parser.add_argument("--start-at", type=int)
    parser.add_argument("--only-timeouts", action='store_true')
    parser.add_argument("--max-trials", "-t", type=int, default=None)
    parser.add_argument("--verbose", '-v', action="count", default=0)

    args = parser.parse_args()

    planning_results_dir = args.ift_dir / 'planning_results'
    for i, results_dir in enumerate(sorted(planning_results_dir.iterdir())):
        if args.start_at is not None and i < args.start_at:
            continue

        if not results_dir.is_dir():
            continue

        print(Style.BRIGHT + Fore.GREEN + results_dir.name + Style.RESET_ALL)
        scenario, metadata = results_utils.get_scenario_and_metadata(results_dir)
        classifier_params = classifier_params_from_planner_params(metadata['planner_params'])
        if args.threshold is None:
            threshold = classifier_params['classifier_dataset_hparams']['labeling_params']['threshold']
        else:
            threshold = args.threshold

        for trial_idx, datum, datum_filename in results_utils.trials_generator(results_dir):
            if args.max_trials is not None and trial_idx >= args.max_trials:
                continue

            should_skip = args.only_timeouts and datum['trial_status'] != TrialStatus.Timeout
            if should_skip:
                continue

            msg = f"trial {trial_idx}, status {datum['trial_status']}"
            print(Fore.LIGHTBLUE_EX + msg)
            recorder = ScreenRecorder(outdir=datum_filename.parent) if args.record else None
            plot_steps(scenario, datum, metadata, {'threshold': threshold}, args.verbose, args.full_plan, recorder)


if __name__ == '__main__':
    main()
