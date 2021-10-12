#!/usr/bin/env python
import argparse
import pathlib

import colorama
from colorama import Style, Fore

from analysis import results_utils
from analysis.results_utils import classifier_params_from_planner_params, plot_steps
from arc_utilities import ros_init
from link_bot_planning.trial_result import TrialStatus
from link_bot_pycommon.screen_recorder import ScreenRecorder
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


def trials_gen(planning_results_dir, args):
    for results_dir in planning_results_dir:
        if not results_dir.is_dir():
            continue
        scenario, metadata = results_utils.get_scenario_and_metadata(results_dir)
        classifier_params = classifier_params_from_planner_params(metadata['planner_params'])
        if args.threshold is None:
            threshold = classifier_params['classifier_dataset_hparams']['labeling_params']['threshold']
        else:
            threshold = args.threshold

        for trial_idx, datum, datum_filename in results_utils.trials_generator(results_dir):

            should_skip = args.only_timeouts and datum['trial_status'] != TrialStatus.Timeout
            if should_skip:
                continue

            yield results_dir, trial_idx, scenario, datum, datum_filename, metadata, threshold


@ros_init.with_ros("plot_ift_results")
def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("ift_dir", type=pathlib.Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--full-plan", action='store_true')
    parser.add_argument("--record", action='store_true')
    parser.add_argument("--only-timeouts", action='store_true')
    parser.add_argument("--verbose", '-v', action="count", default=0)

    args = parser.parse_args()

    planning_results_dir = args.ift_dir / 'planning_results'

    planning_results_dirs = sorted(planning_results_dir.iterdir())

    g = list(trials_gen(planning_results_dirs, args))
    anim = RvizAnimationController(n_time_steps=len(g), ns='trajs')

    while not anim.done:
        j = anim.t()
        results_dir, trial_idx, scenario, datum, datum_filename, metadata, threshold = g[j]
        print(Style.BRIGHT + Fore.GREEN + results_dir.name + Style.RESET_ALL)
        print(Fore.LIGHTBLUE_EX + f"trial {trial_idx}, status {datum['trial_status']}" + Fore.RESET)
        recorder = ScreenRecorder(outdir=datum_filename.parent) if args.record else None
        plot_steps(scenario, datum, metadata, {'threshold': threshold}, args.verbose, args.full_plan, recorder)

        anim.step()


if __name__ == '__main__':
    main()
