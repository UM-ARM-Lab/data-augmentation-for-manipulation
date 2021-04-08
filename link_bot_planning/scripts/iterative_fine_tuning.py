#!/usr/bin/env python
import argparse
import logging
import pathlib
import warnings
from typing import Dict

from link_bot_gazebo import gazebo_services
from link_bot_pycommon.pycommon import pathify

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    from ompl import util as ou

import colorama
import hjson
import tensorflow as tf
from colorama import Fore

import rospy
from arc_utilities import ros_init
from link_bot.link_bot_classifiers.scripts.fine_tune_classifier import fine_tune_classifier
from link_bot.link_bot_planning.scripts.results_to_classifier_dataset import ResultsToClassifierDataset
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
        'test_scenes_dir': test_scenes_dir,
        'checkpoint':      checkpoint.as_posix(),
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
    checkpoint = log['checkpoint']
    test_scenes_dir = log['test_scenes_dir']

    outdir = logfile_name.parent

    job_chunker = JobChunker(logfile_name=logfile_name)
    trials_directory = outdir / 'classifier_training_logdir'

    service_provider = gazebo_services.GazeboServices()

    latest_checkpoint = checkpoint
    fine_tuning_dataset_dirs = []
    for fine_tuning_iteration in range(num_fine_tuning_iterations):
        jobkey = f"iteration {fine_tuning_iteration}"
        job_chunker.setup_key(jobkey)
        sub_job_chunker = job_chunker.sub_chunker(jobkey)
        if sub_job_chunker.is_done():
            print(f"Found results for iteration {fine_tuning_iteration}, continuing")
            continue

        # run planning
        planning_results_dir = outdir / 'planning_results' / f'iteration_{fine_tuning_iteration}_planning'
        planner_params['classifier_model_dir'] = [
            latest_checkpoint,
            pathlib.Path('cl_trials/new_feasibility_baseline/none'),
        ]
        evaluate_planning(planner_params=planner_params,
                          job_chunker=sub_job_chunker,
                          outdir=planning_results_dir,
                          trials=[0],
                          no_execution=no_execution,
                          timeout=timeout,
                          test_scenes_dir=test_scenes_dir,
                          log_full_tree=False,
                          how_to_handle=on_exception,
                          )
        service_provider.pause()

        # results to classifier dataset
        new_dataset_dir = outdir / 'classifier_datasets' / f'iteration_{fine_tuning_iteration}_dataset'
        r = ResultsToClassifierDataset(results_dir=planning_results_dir, outdir=new_dataset_dir)
        r.run()
        fine_tuning_dataset_dirs.append(new_dataset_dir)

        # fine tune (on all of the classifier datasets so far)
        new_latest_checkpoint = fine_tune_classifier(dataset_dirs=fine_tuning_dataset_dirs,
                                                     checkpoint=latest_checkpoint,
                                                     log=f'iteration_{fine_tuning_iteration}_training_logdir',
                                                     trials_directory=trials_directory,
                                                     batch_size=log['batch_size'],
                                                     epochs=log['epochs'])
        latest_checkpoint = new_latest_checkpoint

        fine_tuning_iteration_result = {
            'trial_path': new_latest_checkpoint.as_posix(),

        }
        job_chunker.store_result(jobkey, fine_tuning_iteration_result)

    job_chunker.done()


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
