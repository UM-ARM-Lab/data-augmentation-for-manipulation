#!/usr/bin/env python
import argparse
import logging
import pathlib

import colorama
import tensorflow as tf
from colorama import Fore
from ompl import util as ou

import rospy
from arc_utilities import ros_init
from link_bot.link_bot_classifiers.scripts.fine_tune_classifier import fine_tune_classifier
from link_bot.link_bot_planning.scripts.results_to_classifier_dataset import ResultsToClassifierDataset
from link_bot_data.dataset_utils import data_directory
from link_bot_planning.planning_evaluation import load_planner_params, evaluate_planning
from link_bot_pycommon.args import int_set_arg, my_formatter, run_subparsers
from link_bot_pycommon.job_chunking import JobChunker
from moonshine.filepath_tools import load_params, load_hjson


def iterative_fine_tuning(nickname: str,
                          planner_params: pathlib.Path,
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

    logfile_name = outdir / 'logfile.hjson'
    with logfile_name.open("w") as logfile:
        logfile['nickname'] = nickname
        logfile['planner_params'] = planner_params
        logfile['checkpoint'] = checkpoint.as_posix()

    job_chunker = JobChunker(logfile_name=logfile_name)
    trials_directory = outdir / 'classifier_training_logdir'

    params = load_params(outdir)

    planner_params = load_planner_params(planner_params)

    latest_checkpoint = checkpoint
    fine_tuning_dataset_dirs = []
    for fine_tuning_iteration in range(num_fine_tuning_iterations):
        jobkey = f"iteration {fine_tuning_iteration}"
        job_chunker.setup_key(jobkey)
        sub_job_chunker = job_chunker.sub_chunker(jobkey)
        if job_chunker.result_exists(jobkey):
            print(f"Found results for iteration {fine_tuning_iteration}, continuing")
            continue

        # run planning
        planning_results_dir = outdir / 'planning_results' / f'iteration_{fine_tuning_iteration}_planning'
        planning_results_dir = evaluate_planning(planner_params=planner_params,
                                                 job_chunker=sub_job_chunker,
                                                 outdir=planning_results_dir,
                                                 no_execution=no_execution,
                                                 timeout=timeout,
                                                 test_scenes_dir=test_scenes_dir,
                                                 log_full_tree=False,
                                                 how_to_handle=on_exception,
                                                 )

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
                                                     batch_size=params['batch_size'],
                                                     epochs=params['epochs'])
        latest_checkpoint = new_latest_checkpoint

        fine_tuning_iteration_result = {
            'trial_path': new_latest_checkpoint.as_posix(),

        }
        job_chunker.store_result(jobkey, fine_tuning_iteration_result)


def start_main(args):
    iterative_fine_tuning(nickname=args.nickname,
                          planner_params=args.planner_params,
                          checkpoint=args.checkpoint,
                          num_fine_tuning_iterations=args.num_fine_tuning_iterations,
                          no_execution=args.no_execution,
                          timeout=args.timeout,
                          test_scenes_dir=args.test_scenes_dir,
                          on_exception=args.on_exception,
                          )


def resume_main(args):
    logfile = load_hjson(args.logfile)
    iterative_fine_tuning(nickname=logfile['nickname'],
                          planner_params=logfile['planner_params'],
                          checkpoint=pathlib.Path(logfile['checkpoint']),
                          num_fine_tuning_iterations=args.num_fine_tuning_iterations,
                          no_execution=args.no_execution,
                          timeout=args.timeout,
                          test_scenes_dir=args.test_scenes_dir,
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
