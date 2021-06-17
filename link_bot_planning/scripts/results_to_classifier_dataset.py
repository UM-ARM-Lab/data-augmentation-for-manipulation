#!/usr/bin/env python
import argparse
import pathlib

from arc_utilities import ros_init
from link_bot_planning.results_to_classifier_dataset import ResultsToClassifierDataset
from link_bot_pycommon.args import int_set_arg, BooleanOptionalAction


@ros_init.with_ros("results_to_dataset")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=pathlib.Path, help='directory containing metrics.json')
    parser.add_argument("outdir", type=pathlib.Path, help='output directory')
    parser.add_argument('--full-tree', action=BooleanOptionalAction)
    parser.add_argument("--labeling-params", type=pathlib.Path, help='labeling params')
    parser.add_argument("--visualize", action='store_true', help='visualize')
    parser.add_argument("--regenerate", action='store_true', help='regenerate')
    parser.add_argument("--only-rejected-transitions", action='store_true',
                        help='only keep transitions the planner rejected')
    parser.add_argument("--trial-indices", type=int_set_arg, help='which plan(s) to show')
    parser.add_argument("--save-format", type=str, choices=['pkl', 'tfrecord'], default='pkl')

    args = parser.parse_args()

    r = ResultsToClassifierDataset(results_dir=args.results_dir,
                                   outdir=args.outdir,
                                   labeling_params=args.labeling_params,
                                   trial_indices=args.trial_indices,
                                   visualize=args.visualize,
                                   save_format=args.save_format,
                                   full_tree=args.full_tree,
                                   regenerate=args.regenerate,
                                   only_rejected_transitions=args.only_rejected_transitions,
                                   max_examples_per_trial=None)

    r.run()


if __name__ == '__main__':
    main()
