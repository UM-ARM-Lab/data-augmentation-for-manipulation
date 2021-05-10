#!/usr/bin/env python
import argparse
import pathlib

import colorama

from arc_utilities import ros_init
from link_bot_planning.results_to_recovery_dataset import ResultsToRecoveryDataset
from link_bot_pycommon.args import int_set_arg


@ros_init.with_ros("results_to_recovery_dataset")
def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=pathlib.Path, help='directory containing metrics.json')
    parser.add_argument("outdir", type=pathlib.Path, help='output directory')
    parser.add_argument("--labeling-params", type=pathlib.Path, help='labeling params')
    parser.add_argument("--visualize", action='store_true', help='visualize')
    parser.add_argument("--regenerate", action='store_true', help='ignore existing outputs, overrides existing data')
    parser.add_argument("--trial-indices", type=int_set_arg, help='which plan(s) to show')

    args = parser.parse_args()

    r = ResultsToRecoveryDataset(results_dir=args.results_dir,
                                 outdir=args.outdir,
                                 labeling_params=args.labeling_params,
                                 trial_indices=args.trial_indices,
                                 regenerate=args.regenerate,
                                 visualize=args.visualize)
    r.run()


if __name__ == '__main__':
    main()
