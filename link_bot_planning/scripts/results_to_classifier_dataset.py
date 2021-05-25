#!/usr/bin/env python
import argparse
import pathlib

import colorama
import numpy as np

from arc_utilities import ros_init
from link_bot_planning.results_to_classifier_dataset import ResultsToClassifierDataset
from link_bot_pycommon.args import my_formatter, int_set_arg, BooleanOptionalAction


@ros_init.with_ros("results_to_dataset")
def main():
    colorama.init(autoreset=True)
    np.set_printoptions(linewidth=250, precision=3, suppress=True)
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("results_dir", type=pathlib.Path, help='directory containing metrics.json')
    parser.add_argument("outdir", type=pathlib.Path, help='output directory')
    parser.add_argument('--full-tree', action=BooleanOptionalAction)
    parser.add_argument("--labeling-params", type=pathlib.Path, help='labeling params')
    parser.add_argument("--visualize", action='store_true', help='visualize')
    parser.add_argument("--trial-indices", type=int_set_arg, help='which plan(s) to show')

    args = parser.parse_args()

    r = ResultsToClassifierDataset(args.results_dir,
                                   args.outdir,
                                   args.labeling_params,
                                   args.trial_indices,
                                   args.visualize,
                                   args.full_tree)
    r.run()


if __name__ == '__main__':
    main()
