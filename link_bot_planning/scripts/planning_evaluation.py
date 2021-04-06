#!/usr/bin/env python
import argparse
import logging
import pathlib

import colorama
import tensorflow as tf

from arc_utilities import ros_init
from arc_utilities.algorithms import nested_dict_update
from link_bot_data.dataset_utils import data_directory
from link_bot_planning.planning_evaluation import planning_evaluation
from link_bot_pycommon.args import my_formatter, int_set_arg
from moonshine.filepath_tools import load_hjson


@ros_init.with_ros("planning_evaluation")
def main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)
    tf.autograph.set_verbosity(0)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('planners_params', type=pathlib.Path, nargs='+',
                        help='json file(s) describing what should be compared')
    parser.add_argument("trials", type=int_set_arg, default="0-50")
    parser.add_argument("nickname", type=str, help='used in making the output directory')
    parser.add_argument("--test-scenes-dir", type=pathlib.Path)
    parser.add_argument("--timeout", type=int, help='timeout to override what is in the planner config file')
    parser.add_argument("--seed", type=int, help='an additional seed for testing randomness', default=0)
    parser.add_argument("--no-execution", action="store_true", help='no execution')
    parser.add_argument("--on-exception", choices=['raise', 'catch', 'retry'], default='retry')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument('--record', action='store_true', help='record')
    parser.add_argument('--no-use-gt-rope', action='store_true', help='use ground truth rope state')

    args = parser.parse_args()

    root = data_directory(pathlib.Path('results') / f"{args.nickname}-planning-evaluation")

    planners_params = []
    for planner_params_filename in args.planners_params:
        planner_params = load_planner_params(planner_params_filename)
        planners_params.append((planner_params_filename.stem, planner_params))

    planning_evaluation(outdir=root,
                        planners_params=planners_params,
                        trials=args.trials,
                        how_to_handle=args.on_exception,
                        use_gt_rope=not args.no_use_gt_rope,
                        verbose=args.verbose,
                        timeout=args.timeout,
                        test_scenes_dir=args.test_scenes_dir,
                        no_execution=args.no_execution,
                        logfile_name=None,
                        record=args.record,
                        seed=args.seed,
                        )


def load_planner_params(filename: pathlib.Path):
    top_level_common_filename = filename.parent.parent / 'common.hjson'
    top_level_common_params = load_hjson(top_level_common_filename)

    common_filename = filename.parent / 'common.hjson'
    common_params = load_hjson(common_filename)

    params = load_hjson(filename)
    common_params = nested_dict_update(top_level_common_params, common_params)
    params = nested_dict_update(common_params, params)
    return params


if __name__ == '__main__':
    main()
