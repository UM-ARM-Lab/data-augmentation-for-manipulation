#!/usr/bin/env python
import argparse
import logging
import pathlib

import tensorflow as tf

from arc_utilities import ros_init
from link_bot_planning.planning_evaluation import load_planner_params, evaluate_planning
from link_bot_planning.results_to_dynamics_dataset import ResultsToDynamicsDataset
from link_bot_planning.test_scenes import get_all_scene_indices
from link_bot_pycommon.args import int_set_arg
from link_bot_pycommon.job_chunking import JobChunker
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


@ros_init.with_ros("collect_dynamics_data_planning")
def main():
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('planner_params', type=pathlib.Path, help='planner params hjson file')
    parser.add_argument('classifier', type=pathlib.Path)
    parser.add_argument("test_scenes_dir", type=pathlib.Path)
    parser.add_argument("nickname", type=pathlib.Path, help='used in making the output directory')
    parser.add_argument("--trials", type=int_set_arg)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=int, help='timeout to override what is in the planner config file')
    parser.add_argument("--on-exception", choices=['raise', 'catch', 'retry'], default='retry')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")

    args = parser.parse_args()

    planning_outdir = pathlib.Path("results") / args.nickname

    planner_params = load_planner_params(args.planner_params)
    planner_params['method_name'] = args.outdir.name
    planner_params["classifier_model_dir"] = [args.classifier, pathlib.Path("cl_trials/new_feasibility_baseline/none")]

    if not args.test_scenes_dir.exists():
        print(f"Test scenes dir {args.test_scenes_dir} does not exist")
        return

    if args.trials is None:
        args.trials = list(get_all_scene_indices(args.test_scenes_dir))
        print(args.trials)

    job_chunker = JobChunker(planning_outdir / f'logfile.hjson')
    evaluate_planning(planner_params=planner_params,
                      job_chunker=job_chunker,
                      trials=args.trials,
                      outdir=planning_outdir,
                      verbose=args.verbose,
                      timeout=args.timeout,
                      test_scenes_dir=args.test_scenes_dir,
                      seed=args.seed,
                      how_to_handle=args.on_exception)

    dynamics_outdir = pathlib.Path('fwd_model_data') / args.nickname
    r = ResultsToDynamicsDataset(results_dir=planning_outdir, outdir=dynamics_outdir, traj_length=args.traj_length)
    r.run()


if __name__ == '__main__':
    main()
