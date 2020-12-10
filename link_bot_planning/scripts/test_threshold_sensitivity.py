#!/usr/bin/env python
import argparse
import logging
import pathlib

import colorama
import hjson
import tensorflow as tf

import rospy
from link_bot_planning.planning_evaluation import planning_evaluation
from link_bot_pycommon.args import my_formatter, int_range_arg


def main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('planner_params', type=pathlib.Path, help='json file describing what should be compared')
    parser.add_argument("trials", type=int_range_arg, default="0-50")
    parser.add_argument("test_scenes_dir", type=pathlib.Path)
    parser.add_argument("classifiers_config", type=pathlib.Path)
    parser.add_argument("logfile", type=pathlib.Path)
    parser.add_argument("--timeout", type=int, help='timeout to override what is in the planner config file')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument('--start-at', type=int, default=0)
    parser.add_argument('--stop-at', type=int, default=-1)
    parser.add_argument('--use-gt-rope', action='store_true')

    args = parser.parse_args()

    rospy.init_node("test_threshold_sensitivity")

    root = pathlib.Path('results') / f"dragging-test-threshold-sensitivity"

    with args.classifiers_config.open("r") as classifiers_config_file:
        classifiers_config = hjson.load(classifiers_config_file)

    planners_params = []
    for threshold, configs_for_threshold in classifiers_config.items():
        for seed, classifier_model_dir in configs_for_threshold.items():
            planners_params_common_filename = args.planner_params.parent / 'common.hjson'
            with planners_params_common_filename.open('r') as planners_params_common_file:
                planner_params_common_str = planners_params_common_file.read()
            planner_params = hjson.loads(planner_params_common_str)
            with args.planner_params.open('r') as planner_params_file:
                planner_params_str = planner_params_file.read()
            planner_params.update(hjson.loads(planner_params_str))

            classifiers_dir = pathlib.Path('cl_trials/scirob_dragging_test_threshold_sensitivity')
            classifier_model_dir = classifiers_dir / classifier_model_dir / 'best_checkpoint'
            planner_params['classifier_model_dir'] = classifier_model_dir
            method_name = f"threshold={threshold}"
            planner_params['method_name'] = method_name

            planners_params.append((method_name, planner_params))

    return planning_evaluation(outdir=root,
                               planners_params=planners_params,
                               trials=args.trials,
                               use_gt_rope=args.use_gt_rope,
                               start_idx=args.start_at,
                               stop_idx=args.stop_at,
                               verbose=args.verbose,
                               timeout=args.timeout,
                               test_scenes_dir=args.test_scenes_dir,
                               logfile_name=args.logfile)


if __name__ == '__main__':
    main()
