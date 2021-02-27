#!/usr/bin/env python
import argparse
import logging
import pathlib

import colorama
import hjson
import numpy as np
import tensorflow as tf

from arc_utilities import ros_init
from link_bot_data.dataset_utils import data_directory
from link_bot_planning.planning_evaluation import planning_evaluation
from link_bot_pycommon.args import my_formatter, int_set_arg


@ros_init.with_ros("planning_evaluation")
def main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)
    tf.autograph.set_verbosity(0)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('planner_params', type=pathlib.Path, help='planner params file')
    parser.add_argument("trials", type=int_set_arg, default="0-50")
    parser.add_argument("nickname", type=str, help='used in making the output directory')
    parser.add_argument("--test-scenes-dir", type=pathlib.Path)
    parser.add_argument("--timeout", type=int, help='timeout to override what is in the planner config file')
    parser.add_argument("--seed", type=int, help='an additional seed for testing randomness', default=0)
    parser.add_argument("--no-execution", action="store_true", help='no execution')
    parser.add_argument("--on-exception", choices=['raise', 'catch', 'retry'], default='retry')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument('--record', action='store_true', help='record')

    args = parser.parse_args()

    root = data_directory(pathlib.Path('results') / f"{args.nickname}-planning-evaluation")

    planners_params = []
    for cmp in np.linspace(0, 1, 8):
        planners_params_common_filename = args.planner_params.parent / 'common.hjson'
        with planners_params_common_filename.open('r') as planners_params_common_file:
            planner_params_common_str = planners_params_common_file.read()
        planner_params = hjson.loads(planner_params_common_str)
        with args.planner_params.open('r') as planner_params_file:
            planner_params_str = planner_params_file.read()
        planner_params.update(hjson.loads(planner_params_str))
        planner_params['classifier_mistake_probability'] = cmp
        method_name = planner_params['method_name'] + f" {cmp=}"
        planner_params['method_name'] = method_name.replace(" ", "-")
        planners_params.append((method_name, planner_params))

    planning_evaluation(outdir=root,
                        planners_params=planners_params,
                        trials=args.trials,
                        on_exception=args.on_exception,
                        use_gt_rope=True,
                        verbose=args.verbose,
                        timeout=args.timeout,
                        test_scenes_dir=args.test_scenes_dir,
                        no_execution=args.no_execution,
                        logfile_name=None,
                        record=args.record,
                        seed=args.seed,
                        log_full_tree=False,
                        )


if __name__ == '__main__':
    main()
