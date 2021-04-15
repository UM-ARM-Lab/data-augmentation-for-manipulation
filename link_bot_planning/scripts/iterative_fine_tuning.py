#!/usr/bin/env python
import argparse
import itertools
import logging
import pathlib
import warnings
from dataclasses import dataclass
from typing import Dict, List

from link_bot_classifiers.fine_tune_classifier import fine_tune_classifier
from link_bot_gazebo.gazebo_services import get_gazebo_processes
from link_bot_planning.analysis.results_metrics import load_analysis_params, generate_per_trial_metrics, Successes
from link_bot_planning.results_to_classifier_dataset import ResultsToClassifierDataset
from link_bot_planning.test_scenes import get_all_scene_indices
from link_bot_pycommon import notifyme
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
from link_bot_pycommon.args import my_formatter, run_subparsers
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
    from_env, to_env = nickname.split("_to_")

    logfile_name = outdir / 'logfile.hjson'
    log = {
        'nickname':        nickname,
        'planner_params':  planner_params,
        'test_scenes_dir': test_scenes_dir.as_posix(),
        'checkpoints':     [checkpoint.as_posix()],
        'batch_size':      8,
        'epochs':          25,
        'early_stopping':  True,
        'from_env':        from_env,
        'to_env':          to_env,
    }
    with logfile_name.open("w") as logfile:
        hjson.dump(log, logfile)

    ift = IterativeFineTuning(log=log,
                              no_execution=no_execution,
                              timeout=timeout,
                              on_exception=on_exception,
                              logfile_name=logfile_name,
                              )
    ift.run(num_fine_tuning_iterations=num_fine_tuning_iterations)


@dataclass
class IterationData:
    fine_tuning_dataset_dirs: List[pathlib.Path]
    fine_tuning_iteration: int
    iteration_chunker: JobChunker
    latest_checkpoint_dir: pathlib.Path


class IterativeFineTuning:

    def __init__(self,
                 log: Dict,
                 no_execution: bool,
                 timeout: int,
                 on_exception: str,
                 logfile_name: pathlib.Path,
                 ):
        self.no_execution = no_execution
        self.on_exception = on_exception
        self.timeout = timeout
        self.log = log
        self.planner_params = pathify(self.log['planner_params'])
        self.test_scenes_dir = pathlib.Path(self.log['test_scenes_dir'])

        self.gazebo_processes = get_gazebo_processes()

        self.outdir = logfile_name.parent

        self.job_chunker = JobChunker(logfile_name=logfile_name)
        self.trials_directory = self.outdir / 'classifier_training_logdir'
        self.planning_results_root_dir = self.outdir / 'planning_results'

        self.trial_idx_gen = itertools.cycle(get_all_scene_indices(self.test_scenes_dir))

    def run(self, num_fine_tuning_iterations: int):
        checkpoints = paths_from_json(self.log['checkpoints'])
        latest_checkpoint_dir = checkpoints[-1]
        fine_tuning_dataset_dirs = []
        for fine_tuning_iteration in range(num_fine_tuning_iterations):
            jobkey = f"iteration {fine_tuning_iteration}"
            iteration_chunker = self.job_chunker.sub_chunker(jobkey)
            latest_success_rate = iteration_chunker.get_result('latest_success_rate')

            iteration_data = IterationData(fine_tuning_dataset_dirs=fine_tuning_dataset_dirs,
                                           fine_tuning_iteration=fine_tuning_iteration,
                                           iteration_chunker=iteration_chunker,
                                           latest_checkpoint_dir=latest_checkpoint_dir,
                                           )

            # planning
            latest_success_rate, planning_results_dir = self.plan_and_execute(iteration_data, latest_success_rate)

            # convert results to classifier dataset
            self.update_datasets(iteration_data, planning_results_dir)

            # fine tune (on all of the classifier datasets so far)
            latest_checkpoint_dir = self.fine_tune(iteration_data, latest_checkpoint_dir)

        [p.kill() for p in self.gazebo_processes]

    def plan_and_execute(self, iteration_data: IterationData, latest_success_rate):
        i = iteration_data.fine_tuning_iteration
        trial_idx = next(self.trial_idx_gen)
        planning_chunker = iteration_data.iteration_chunker.sub_chunker('planning')
        planning_results_dir = pathify(planning_chunker.get_result('planning_results_dir'))
        if planning_results_dir is None:
            planning_results_dir = self.planning_results_root_dir / f'iteration_{i:02d}_planning'
            latest_checkpoint = iteration_data.latest_checkpoint_dir / 'best_checkpoint'
            self.planner_params['classifier_model_dir'] = [
                latest_checkpoint,
                pathlib.Path('cl_trials/new_feasibility_baseline/none'),
            ]
            self.planner_params['fine_tuning_iteration'] = i

            [p.resume() for p in self.gazebo_processes]
            evaluate_planning(planner_params=self.planner_params,
                              trials=[trial_idx],
                              job_chunker=planning_chunker,
                              outdir=planning_results_dir,
                              no_execution=self.no_execution,
                              timeout=self.timeout,
                              test_scenes_dir=self.test_scenes_dir,
                              log_full_tree=False,
                              how_to_handle=self.on_exception,
                              verbose=-1,
                              )
            [p.suspend() for p in self.gazebo_processes]

            analysis_params = load_analysis_params()
            metrics = generate_per_trial_metrics(analysis_params=analysis_params,
                                                 subfolders_ordered=[planning_results_dir],
                                                 method_names=[self.planner_params['method_name']])
            successes = metrics[Successes].values[self.planner_params['method_name']]
            latest_success_rate = successes.sum() / successes.shape[0]
        print(Fore.CYAN + f"Iteration {i} {latest_success_rate * 100:.1f}%")
        return planning_results_dir

    def update_datasets(self, iteration_data: IterationData, planning_results_dir):
        i = iteration_data.fine_tuning_iteration
        dataset_chunker = iteration_data.iteration_chunker.sub_chunker('dataset')
        new_dataset_dir = pathify(dataset_chunker.get_result('new_dataset_dir'))
        if new_dataset_dir is None:
            new_dataset_dir = self.outdir / 'classifier_datasets' / f'iteration_{i}_dataset'
            r = ResultsToClassifierDataset(results_dir=planning_results_dir, outdir=new_dataset_dir, verbose=-1)
            r.run()
            dataset_chunker.store_result('new_dataset_dir', new_dataset_dir.as_posix())
        iteration_data.fine_tuning_dataset_dirs.append(new_dataset_dir)

    def fine_tune(self, iteration_data: IterationData, latest_checkpoint_dir: pathlib.Path):
        i = iteration_data.fine_tuning_iteration
        latest_checkpoint = iteration_data.latest_checkpoint_dir / 'best_checkpoint'
        fine_tune_chunker = iteration_data.iteration_chunker.sub_chunker('fine tune')
        new_latest_checkpoint_dir = pathify(fine_tune_chunker.get_result('new_latest_checkpoint_dir'))
        if new_latest_checkpoint_dir is None:
            new_latest_checkpoint_dir = fine_tune_classifier(dataset_dirs=iteration_data.fine_tuning_dataset_dirs,
                                                             checkpoint=latest_checkpoint,
                                                             log=f'iteration_{i}_training_logdir',
                                                             trials_directory=self.trials_directory,
                                                             batch_size=self.log['batch_size'],
                                                             early_stopping=self.log['early_stopping'],
                                                             epochs=self.log['epochs'])
            fine_tune_chunker.store_result('new_latest_checkpoint_dir', new_latest_checkpoint_dir.as_posix())
        print(Fore.CYAN + f"Finished iteration {i}")
        return latest_checkpoint_dir


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
    ift = IterativeFineTuning(log=log,
                              logfile_name=args.logfile,
                              no_execution=args.no_execution,
                              timeout=args.timeout,
                              on_exception=args.on_exception,
                              )
    ift.run(num_fine_tuning_iterations=args.n_iters)


def add_args(start_parser):
    start_parser.add_argument("--timeout", type=int, help='timeout to override what is in the planner config file')
    start_parser.add_argument("--seed", type=int, help='an additional seed for testing randomness', default=0)
    start_parser.add_argument("--n-iters", '-n', type=int, help='number of iterations of fine tuning', default=500)
    start_parser.add_argument("--no-execution", action="store_true", help='no execution')
    start_parser.add_argument("--on-exception", choices=['raise', 'catch', 'retry'], default='retry')
    start_parser.add_argument('--verbose', '-v', action='count', default=0,
                              help="use more v's for more verbose, like -vvv")


@ros_init.with_ros("iterative_fine_tuning")
@notifyme.notify()
def ift_main():
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
    ift_main()
