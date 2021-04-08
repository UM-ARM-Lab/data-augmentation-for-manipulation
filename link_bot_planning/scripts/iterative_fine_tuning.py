#!/usr/bin/env python
import argparse
import logging
import pathlib
import warnings
from typing import Dict

from link_bot_classifiers.fine_tune_classifier import fine_tune_classifier
from link_bot_gazebo.gazebo_services import get_gazebo_processes
from link_bot_planning.results_metrics import load_analysis_params, generate_metrics, PercentageSuccess
from link_bot_planning.results_to_classifier_dataset import ResultsToClassifierDataset
from link_bot_pycommon.pycommon import pathify, paths_from_json

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    from ompl import util as ou

import colorama
import hjson
import tensorflow as tf
from colorama import Fore

import rospy
from arc_utilities import ros_init
from link_bot_data.dataset_utils import data_directory
from link_bot_planning.planning_evaluation import load_planner_params, evaluate_planning
from link_bot_pycommon.args import int_set_arg, my_formatter, run_subparsers
from link_bot_pycommon.job_chunking import JobChunker
from moonshine.filepath_tools import load_hjson


def start_iterative_fine_tuning(nickname: str,
                                planner_params_filename: pathlib.Path,
                                checkpoint: pathlib.Path,
                                num_fine_tuning_iterations: int,
                                no_execution: bool,
                                timeout: int,
                                test_scenes_dir: pathlib.Path,
                                on_exception: str,
                                ):
    # setup
    outdir = data_directory(pathlib.Path('results') / 'iterative_fine_tuning' / f"{nickname}")

    if not outdir.exists():
        rospy.loginfo(Fore.YELLOW + "Creating output directory: {}".format(outdir))
        outdir.mkdir(parents=True)

    planner_params = load_planner_params(planner_params_filename)

    logfile_name = outdir / 'logfile.hjson'
    log = {
        'nickname':        nickname,
        'planner_params':  planner_params,
        'test_scenes_dir': test_scenes_dir.as_posix(),
        'checkpoints':     [checkpoint.as_posix()],
        'batch_size':      16,
        'epochs':          25,
    }
    with logfile_name.open("w") as logfile:
        hjson.dump(log, logfile)

    iterative_fine_tuning(log=log,
                          num_fine_tuning_iterations=num_fine_tuning_iterations,
                          no_execution=no_execution,
                          timeout=timeout,
                          on_exception=on_exception,
                          logfile_name=logfile_name,
                          )


def iterative_fine_tuning(log: Dict,
                          num_fine_tuning_iterations: int,
                          no_execution: bool,
                          timeout: int,
                          on_exception: str,
                          logfile_name: pathlib.Path,
                          ):
    planner_params = pathify(log['planner_params'])
    checkpoints = paths_from_json(log['checkpoints'])
    test_scenes_dir = pathlib.Path(log['test_scenes_dir'])

    gazebo_processes = get_gazebo_processes()

    outdir = logfile_name.parent

    job_chunker = JobChunker(logfile_name=logfile_name)
    trials_directory = outdir / 'classifier_training_logdir'

    latest_checkpoint_dir = checkpoints[-1]
    fine_tuning_dataset_dirs = []
    latest_success_rate = -1

    for fine_tuning_iteration in range(num_fine_tuning_iterations):
        jobkey = f"iteration {fine_tuning_iteration}"
        iteration_chunker = job_chunker.sub_chunker(jobkey)

        latest_checkpoint = latest_checkpoint_dir / 'best_checkpoint'

        # planning
        planning_chunker = iteration_chunker.sub_chunker('planning')
        planning_results_dir = pathify(planning_chunker.get_result('planning_results_dir'))
        if planning_results_dir is None:
            planning_results_dir = outdir / 'planning_results' / f'iteration_{fine_tuning_iteration}_planning'
            planner_params['classifier_model_dir'] = [
                latest_checkpoint,
                pathlib.Path('cl_trials/new_feasibility_baseline/none'),
            ]

            [p.resume() for p in gazebo_processes]
            evaluate_planning(planner_params=planner_params,
                              job_chunker=planning_chunker,
                              # REMOVE ME!
                              # trials=[0, 1],
                              outdir=planning_results_dir,
                              no_execution=no_execution,
                              timeout=timeout,
                              test_scenes_dir=test_scenes_dir,
                              log_full_tree=False,
                              how_to_handle=on_exception,
                              )
            [p.suspend() for p in gazebo_processes]

            analysis_params = load_analysis_params()
            metrics = generate_metrics(analysis_params, [planning_results_dir])
            successes = metrics[PercentageSuccess].values[planner_params['method_name']]
            latest_success_rate = successes.sum() / successes.shape[0]

        # convert results to classifier dataset
        dataset_chunker = iteration_chunker.sub_chunker('dataset')
        new_dataset_dir = pathify(dataset_chunker.get_result('new_dataset_dir'))
        if new_dataset_dir is None:
            new_dataset_dir = outdir / 'classifier_datasets' / f'iteration_{fine_tuning_iteration}_dataset'
            r = ResultsToClassifierDataset(results_dir=planning_results_dir, outdir=new_dataset_dir)
            r.run()
            dataset_chunker.store_result('new_dataset_dir', new_dataset_dir.as_posix())
        fine_tuning_dataset_dirs.append(new_dataset_dir)

        # fine tune (on all of the classifier datasets so far)
        fine_tune_chunker = iteration_chunker.sub_chunker('fine tune')
        new_latest_checkpoint_dir = pathify(fine_tune_chunker.get_result('new_latest_checkpoint_dir'))
        if new_latest_checkpoint_dir is None:
            new_latest_checkpoint_dir = fine_tune_classifier(dataset_dirs=fine_tuning_dataset_dirs,
                                                             checkpoint=latest_checkpoint,
                                                             log=f'iteration_{fine_tuning_iteration}_training_logdir',
                                                             trials_directory=trials_directory,
                                                             batch_size=log['batch_size'],
                                                             epochs=log['epochs'])
            fine_tune_chunker.store_result('new_latest_checkpoint_dir', new_latest_checkpoint_dir.as_posix())

        latest_checkpoint_dir = new_latest_checkpoint_dir
        print(Fore.CYAN + f"Finished iteration {fine_tuning_iteration} {latest_success_rate * 100:.1f}%")

    [p.kill() for p in gazebo_processes]


def start_main(args):
    start_iterative_fine_tuning(nickname=args.nickname,
                                planner_params_filename=args.planner_params,
                                checkpoint=args.checkpoint,
                                num_fine_tuning_iterations=args.n_iters,
                                no_execution=args.no_execution,
                                timeout=args.timeout,
                                test_scenes_dir=args.test_scenes_dir,
                                on_exception=args.on_exception,
                                )


def resume_main(args):
    log = load_hjson(args.logfile)
    iterative_fine_tuning(log=log,
                          logfile_name=args.logfile,
                          num_fine_tuning_iterations=args.n_iters,
                          no_execution=args.no_execution,
                          timeout=args.timeout,
                          on_exception=args.on_exception,
                          )


def add_args(start_parser):
    start_parser.add_argument("--trials", type=int_set_arg, default="0-29")
    start_parser.add_argument("--timeout", type=int, help='timeout to override what is in the planner config file')
    start_parser.add_argument("--seed", type=int, help='an additional seed for testing randomness', default=0)
    start_parser.add_argument("--n-iters", '-n', type=int, help='number of iterations of fine tuning', default=10)
    start_parser.add_argument("--no-execution", action="store_true", help='no execution')
    start_parser.add_argument("--on-exception", choices=['raise', 'catch', 'retry'], default='retry')
    start_parser.add_argument('--verbose', '-v', action='count', default=0,
                              help="use more v's for more verbose, like -vvv")


@ros_init.with_ros("iterative_fine_tuning")
def main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)
    ou.setLogLevel(ou.LOG_ERROR)
    tf.autograph.set_verbosity(0)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    subparsers = parser.add_subparsers()
    start_parser = subparsers.add_parser('start')
    resume_parser = subparsers.add_parser('resume')

    start_parser.add_argument('planner_params', type=pathlib.Path, help='hjson file from planner_configs/')
    start_parser.add_argument("checkpoint", type=pathlib.Path, help='classifier checkpoint to start from')
    start_parser.add_argument("nickname", type=str, help='used in making the output directory')
    start_parser.add_argument("test_scenes_dir", type=pathlib.Path)
    start_parser.set_defaults(func=start_main)
    add_args(start_parser)

    resume_parser.add_argument("logfile", type=pathlib.Path)
    resume_parser.set_defaults(func=resume_main)
    add_args(resume_parser)

    run_subparsers(parser)


if __name__ == '__main__':
    main()
