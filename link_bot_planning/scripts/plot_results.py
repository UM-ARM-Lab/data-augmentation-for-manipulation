#!/usr/bin/env python
import argparse
import pathlib

from analysis import results_utils
from analysis.results_utils import plot_steps, get_all_results_subdirs, trials_filenames_generator
from arc_utilities import ros_init
from link_bot_gazebo.gazebo_utils import gazebo_suspended
from link_bot_planning.trial_result import TrialStatus
from link_bot_pycommon.serialization import load_gzipped_pickle
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


@ros_init.with_ros("plot_results")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=pathlib.Path, help='directory containing metrics.json')
    parser.add_argument("--full-plan", action='store_true')
    parser.add_argument("--only-timeouts", action='store_true')
    parser.add_argument("--only-reached", action='store_true')
    parser.add_argument("--regenerate", action='store_true')
    parser.add_argument("--verbose", '-v', action="count", default=0)
    parser.add_argument("--threshold", type=float, default=0.06)

    args = parser.parse_args()

    results_dir = get_all_results_subdirs(args.results_dir, regenerate=args.regenerate)[0]
    scenario, metadata = results_utils.get_scenario_and_metadata(results_dir)

    idx_and_filenames = list(trials_filenames_generator(results_dir))

    with gazebo_suspended():
        anim = RvizAnimationController(n_time_steps=len(idx_and_filenames), ns='trajs')

        while not anim.done:
            j = anim.t()
            trial_idx, datum_filename = idx_and_filenames[j]
            datum = load_gzipped_pickle(datum_filename)

            trial_status = datum['trial_status']
            should_skip = (args.only_timeouts and trial_status == TrialStatus.Reached or
                           args.only_reached and trial_status != TrialStatus.Reached)

            if should_skip:
                anim.step()
                continue

            print(f"trial {trial_idx}, status {trial_status}")
            plot_steps(scenario, datum, metadata, {'threshold': args.threshold}, args.verbose, args.full_plan)

            anim.step()


if __name__ == '__main__':
    main()
