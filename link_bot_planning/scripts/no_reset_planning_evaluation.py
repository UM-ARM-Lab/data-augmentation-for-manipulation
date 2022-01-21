#!/usr/bin/env python
import argparse
import logging
import pathlib
import warnings

import colorama
import tensorflow as tf

from arc_utilities import ros_init
from link_bot_data.dataset_utils import make_unique_outdir
from link_bot_planning.no_reset_planning_evaluation import NoResetEvaluatePlanning
from link_bot_planning.planning_evaluation import load_planner_params, evaluate_planning
from link_bot_planning.test_scenes import get_all_scene_indices
from link_bot_pycommon.args import int_set_arg
from link_bot_pycommon.job_chunking import JobChunker
from moonshine.gpu_config import limit_gpu_mem

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    from ompl import util as ou

limit_gpu_mem(None)


@ros_init.with_ros("no_reset_planning_evaluation")
def main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('planner_params', type=pathlib.Path, help='planner params hjson file')
    parser.add_argument("test_scenes_dir", type=pathlib.Path)
    parser.add_argument("outdir", type=pathlib.Path, help='used in making the output directory')
    parser.add_argument("--trials", type=int_set_arg)
    parser.add_argument("--timeout", type=int, help='timeout to override what is in the planner config file')
    parser.add_argument("--seed", type=int, help='an additional seed for testing randomness', default=0)
    parser.add_argument("--no-execution", action="store_true", help='no execution')
    parser.add_argument("--on-exception", choices=['raise', 'catch', 'retry'], default='retry')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument('--record', action='store_true', help='record')
    parser.add_argument('--no-use-gt-rope', action='store_true', help='use ground truth rope state')
    parser.add_argument('--classifier', type=pathlib.Path)

    args = parser.parse_args()

    outdir = make_unique_outdir(args.outdir)

    planner_params = load_planner_params(args.planner_params)
    planner_params['method_name'] = args.outdir.name

    if args.classifier:
        planner_params["classifier_model_dir"] = [args.classifier,
                                                  pathlib.Path("cl_trials/new_feasibility_baseline/none")]

    if not args.test_scenes_dir.exists():
        print(f"Test scenes dir {args.test_scenes_dir} does not exist")
        return

    if args.trials is None:
        args.trials = list(get_all_scene_indices(args.test_scenes_dir))
        print('trials:', args.trials)

    logfile_name = outdir / f'logfile.hjson'
    print(f'logfile: {logfile_name}')
    job_chunker = JobChunker(logfile_name=logfile_name)

    ou.setLogLevel(ou.LOG_ERROR)

    evaluate_planning(planner_params=planner_params,
                      job_chunker=job_chunker,
                      trials=args.trials,
                      outdir=outdir,
                      verbose=args.verbose,
                      record=args.record,
                      no_execution=args.no_execution,
                      timeout=args.timeout,
                      test_scenes_dir=args.test_scenes_dir,
                      seed=args.seed,
                      how_to_handle=args.on_exception,
                      eval_class_type=NoResetEvaluatePlanning,
                      )


if __name__ == '__main__':
    main()
