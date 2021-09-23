#!/usr/bin/env python
import argparse
import itertools
import logging
import pathlib
import warnings
from dataclasses import dataclass
from time import perf_counter
from typing import List
from uuid import uuid4

import numpy as np
from more_itertools import chunked

import link_bot_pycommon.pycommon
from analysis.results_utils import list_all_planning_results_trials
from arc_utilities.algorithms import nested_dict_update
from link_bot_classifiers.fine_tune_classifier import fine_tune_classifier
from link_bot_classifiers.fine_tune_recovery import fine_tune_recovery
from link_bot_gazebo import gazebo_services
from link_bot_gazebo.gazebo_services import get_gazebo_processes
from link_bot_planning.get_planner import get_planner, load_classifier
from link_bot_planning.results_to_classifier_dataset import ResultsToClassifierDataset
from link_bot_planning.results_to_recovery_dataset import ResultsToRecoveryDataset
from link_bot_planning.test_scenes import get_all_scene_indices
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.pycommon import pathify, deal_with_exceptions
from moonshine.gpu_config import limit_gpu_mem

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    from ompl import util as ou

import colorama
import tensorflow as tf
from colorama import Fore, Style

import rospy
from arc_utilities import ros_init
from link_bot_data.load_dataset import compute_batch_size
from link_bot_planning.planning_evaluation import load_planner_params, EvaluatePlanning
from link_bot_pycommon.job_chunking import JobChunker
from moonshine.filepath_tools import load_hjson

limit_gpu_mem(None)


@dataclass
class IterationData:
    iteration: int
    iteration_chunker: JobChunker
    fine_tuning_classifier_dataset_dirs: List[pathlib.Path]
    fine_tuning_recovery_dataset_dirs: List[pathlib.Path]
    latest_classifier_checkpoint_dir: pathlib.Path
    latest_recovery_checkpoint_dir: pathlib.Path


class IterativeFineTuning:

    def __init__(self,
                 outdir: pathlib.Path,
                 on_exception: str,
                 no_execution: bool = True,
                 timeout: int = None,
                 ):
        self.outdir = outdir
        self.no_execution = no_execution
        self.on_exception = on_exception
        self.log_full_tree = False
        self.verbose = -1

        logfile_name = outdir / 'logfile.hjson'
        self.outdir.mkdir(exist_ok=True, parents=True)
        rospy.loginfo(Fore.YELLOW + "Output directory: {}".format(self.outdir))

        self.job_chunker = JobChunker(logfile_name)

        ift_config_filename = pathify(self.job_chunker.load_prompt('ift_config_filename'))
        self.seed = int(self.job_chunker.load_prompt('seed'))
        default_classifier_checkpoint = '/media/shared/cl_trials/untrained-1/August_13_17-03-09_45c09348d1'
        self.initial_classifier_checkpoint = pathify(
            self.job_chunker.load_prompt('initial_classifier_checkpoint', default_classifier_checkpoint))
        self.initial_recovery_checkpoint = pathify(self.job_chunker.load_prompt('initial_recovery_checkpoint', None))
        planner_params_filename = pathify(
            self.job_chunker.load_prompt('planner_params_filename', 'planner_configs/val_car/random_recovery.hjson'))
        self.test_scenes_dir = pathify(
            self.job_chunker.load_prompt('test_scenes_dir', 'test_scenes/swap_straps_no_recovery3'))
        self.test_scenes_indices = pathify(self.job_chunker.load_prompt('test_scenes_indices', None))

        self.ift_config = load_hjson(ift_config_filename)
        self.checkpoint_suffix = 'latest_checkpoint'
        self.ift_config['checkpoint_suffix'] = self.checkpoint_suffix
        self.job_chunker.store_results(self.ift_config)
        self.initial_planner_params = load_planner_params(planner_params_filename)

        self.job_chunker.store_result('from_env', 'untrained')
        self.job_chunker.store_result('to_env', 'car3')

        self.ift_uuid = uuid4()
        self.job_chunker.store_result('ift_uuid', str(self.ift_uuid))
        self.initial_planner_params["log_full_tree"] = self.log_full_tree
        self.initial_planner_params['classifier_model_dir'] = []  # this gets replace at every iteration
        self.tpi = int(self.ift_config['trials_per_iteration'])
        self.classifier_labeling_params = load_hjson(pathlib.Path('labeling_params/classifier/dual.hjson'))
        self.classifier_labeling_params = nested_dict_update(self.classifier_labeling_params,
                                                             self.ift_config.get('labeling_params_update', {}))
        self.recovery_labeling_params = load_hjson(pathlib.Path('labeling_params/recovery/dual.json'))
        self.recovery_labeling_params = nested_dict_update(self.recovery_labeling_params,
                                                           self.ift_config.get('labeling_params_update', {}))
        self.initial_planner_params = nested_dict_update(self.initial_planner_params,
                                                         self.ift_config.get('planner_params_update', {}))
        self.pretraining_config = self.ift_config.get('pretraining', {})

        if timeout is not None:
            rospy.loginfo(f"Overriding with timeout {timeout}")
            self.initial_planner_params["termination_criteria"]['timeout'] = timeout
            self.initial_planner_params["termination_criteria"]['total_timeout'] = timeout

        self.gazebo_processes = get_gazebo_processes()

        self.trials_directory = self.outdir / 'training_logdir'
        self.planning_results_root_dir = self.outdir / 'planning_results'

        if self.test_scenes_indices is None:
            all_trial_indices = list(get_all_scene_indices(self.test_scenes_dir))
        else:
            all_trial_indices = self.test_scenes_indices
        trials_generator_type = self.ift_config['trials_generator_type']
        if trials_generator_type == 'cycle':
            self.trial_indices_generator = chunked(itertools.cycle(all_trial_indices), self.tpi)
        elif trials_generator_type == 'random':
            def _random():
                rng = np.random.RandomState(self.seed)
                while True:
                    yield rng.choice(all_trial_indices, size=self.tpi, replace=False)

            self.trial_indices_generator = _random()
        else:
            raise NotImplementedError(f"Unimplemented {trials_generator_type}")

        # Start Services
        [p.resume() for p in self.gazebo_processes]
        self.service_provider = gazebo_services.GazeboServices()
        self.service_provider.play()  # time needs to be advancing while we setup the planner

        # Setup scenario
        self.scenario = get_scenario(self.initial_planner_params["scenario"])
        self.scenario.on_before_get_state_or_execute_action()
        self.service_provider.setup_env(verbose=self.verbose,
                                        real_time_rate=self.initial_planner_params['real_time_rate'],
                                        max_step_size=0.01,  # FIXME:
                                        play=True)

        self.planner = get_planner(planner_params=self.initial_planner_params,
                                   verbose=self.verbose,
                                   log_full_tree=self.log_full_tree,
                                   scenario=self.scenario)

    def run(self, n_iters: int):
        initial_classifier_checkpoint = pathify(self.job_chunker.get_result('initial_classifier_checkpoint'))
        initial_recovery_checkpoint = pathify(self.job_chunker.get_result('initial_recovery_checkpoint'))

        # NOTE: should I append the pretraining datasets to this? Possibly...
        fine_tuning_classifier_dataset_dirs = []
        fine_tuning_recovery_dataset_dirs = []

        latest_classifier_checkpoint_dir = initial_classifier_checkpoint
        latest_recovery_checkpoint_dir = initial_recovery_checkpoint
        for iteration_idx in range(n_iters):
            jobkey = f"iteration {iteration_idx}"
            iteration_chunker = self.job_chunker.sub_chunker(jobkey)
            iteration_start_time = iteration_chunker.get_result('start_time')
            if iteration_start_time is None:
                iteration_start_time = perf_counter()
                iteration_chunker.store_result('start_time', iteration_start_time)
            iteration_data = IterationData(fine_tuning_classifier_dataset_dirs=fine_tuning_classifier_dataset_dirs,
                                           fine_tuning_recovery_dataset_dirs=fine_tuning_recovery_dataset_dirs,
                                           iteration=iteration_idx,
                                           iteration_chunker=iteration_chunker,
                                           latest_classifier_checkpoint_dir=latest_classifier_checkpoint_dir,
                                           latest_recovery_checkpoint_dir=latest_recovery_checkpoint_dir,
                                           )
            # planning
            planning_results_dir = self.plan_and_execute(iteration_data)

            # convert results to classifier dataset
            new_classifier_dataset_dir, new_recovery_dataset_dir = self.update_datasets(iteration_data,
                                                                                        planning_results_dir)

            iteration_data.fine_tuning_classifier_dataset_dirs.append(new_classifier_dataset_dir)
            iteration_data.fine_tuning_recovery_dataset_dirs.append(new_recovery_dataset_dir)

            # fine tune (on all of the classifier datasets so far)
            # these variables will be used to create the new IterationData
            latest_classifier_checkpoint_dir, latest_recovery_checkpoint_dir = self.fine_tune(iteration_data)

            iteration_end_time = iteration_chunker.get_result('end_time')
            if iteration_end_time is None:
                iteration_end_time = perf_counter()
                iteration_chunker.store_result('end_time', iteration_end_time)
            iteration_time = iteration_end_time - iteration_start_time
            end_iter_msg = f"Finished iteration {iteration_idx}/{n_iters}, {iteration_time:.1f}s"
            print(Style.BRIGHT + end_iter_msg + Style.RESET_ALL)

        [p.kill() for p in self.gazebo_processes]

    def plan_and_execute(self, iteration_data: IterationData):
        i = iteration_data.iteration
        # always last the best checkpoint at iteration 0, that's the pretrained model
        checkpoint_suffix = self.checkpoint_suffix if i != 0 else 'best_checkpoint'

        trials = next(self.trial_indices_generator)
        planning_chunker = iteration_data.iteration_chunker.sub_chunker('planning')
        planning_results_dir = pathify(planning_chunker.get_result('planning_results_dir'))
        if planning_results_dir is None:
            planning_results_dir = self.planning_results_root_dir / f'iteration_{i:04d}_planning'
            latest_classifier_checkpoint = iteration_data.latest_classifier_checkpoint_dir / checkpoint_suffix
            planner_params = self.initial_planner_params.copy()
            if iteration_data.latest_recovery_checkpoint_dir is not None:
                latest_recovery_checkpoint = iteration_data.latest_recovery_checkpoint_dir / 'best_checkpoint'
                planner_params['recovery']['recovery_model_dir'] = latest_recovery_checkpoint
            planner_params['classifier_model_dir'] = [
                latest_classifier_checkpoint,
                pathlib.Path('/media/shared/cl_trials/new_feasibility_baseline/none'),
            ]
            self.initial_planner_params['fine_tuning_iteration'] = i

            # planning evaluation
            [p.resume() for p in self.gazebo_processes]
            classifier_models = load_classifier(planner_params, self.scenario)
            self.planner.classifier_models = classifier_models

            # Use this to pass more info into the results metadata.hjson
            metadata_update = {
                'ift_iteration': iteration_data.iteration,
                'ift_uuid':      self.ift_uuid,
                'ift_config':    self.ift_config,
            }
            runner = EvaluatePlanning(planner=self.planner,
                                      service_provider=self.service_provider,
                                      job_chunker=planning_chunker,
                                      verbose=self.verbose,
                                      planner_params=planner_params,
                                      outdir=planning_results_dir,
                                      trials=trials,
                                      test_scenes_dir=self.test_scenes_dir,
                                      seed=self.seed,
                                      metadata_update=metadata_update)

            deal_with_exceptions(how_to_handle=self.on_exception, function=runner.run)
            [p.suspend() for p in self.gazebo_processes]

        print(Fore.CYAN + f"Iteration {i}")
        return planning_results_dir

    def update_datasets(self, iteration_data: IterationData, planning_results_dir):
        new_classifier_dataset_dir = self.update_classifier_datasets(iteration_data, planning_results_dir)
        if self.ift_config['fine_tune_recovery'] is None:
            new_recovery_dataset_dir = None
        else:
            new_recovery_dataset_dir = self.update_recovery_datasets(iteration_data, planning_results_dir)
        return new_classifier_dataset_dir, new_recovery_dataset_dir

    def update_classifier_datasets(self, iteration_data: IterationData, planning_results_dir):
        i = iteration_data.iteration
        dataset_chunker = iteration_data.iteration_chunker.sub_chunker('classifier dataset')
        new_dataset_dir = pathify(dataset_chunker.get_result('new_dataset_dir'))
        if new_dataset_dir is None:
            print("Updating Classifier Dataset")
            [p.suspend() for p in self.gazebo_processes]

            new_dataset_dir = self.outdir / 'classifier_datasets' / f'iteration_{i:04d}_dataset'
            trial_indices = None
            max_trials = self.ift_config['results_to_classifier_dataset'].get('max_trials', None)
            if max_trials is not None:
                print(Fore.GREEN + f"Using only {max_trials}/{self.tpi} trials for learning" + Fore.RESET)
                filenames = list_all_planning_results_trials(planning_results_dir)
                trial_indices = [i for (i, _) in filenames][:max_trials]
            r = ResultsToClassifierDataset(results_dir=planning_results_dir,
                                           outdir=new_dataset_dir,
                                           labeling_params=self.classifier_labeling_params,
                                           verbose=self.verbose,
                                           trial_indices=trial_indices,
                                           fwd_model=self.planner.fwd_model,
                                           **self.ift_config['results_to_classifier_dataset'])
            r.run()
            new_dataset_dir_rel = new_dataset_dir.relative_to(self.outdir)
            dataset_chunker.store_result('new_dataset_dir', new_dataset_dir_rel.as_posix())
        else:
            new_dataset_dir = self.outdir / new_dataset_dir
        return new_dataset_dir

    def update_recovery_datasets(self, iteration_data: IterationData, planning_results_dir):
        i = iteration_data.iteration
        dataset_chunker = iteration_data.iteration_chunker.sub_chunker('recovery dataset')
        new_dataset_dir = pathify(dataset_chunker.get_result('new_dataset_dir'))
        if new_dataset_dir is None:
            print("Updating Recovery Dataset")
            [p.suspend() for p in self.gazebo_processes]

            new_dataset_dir = self.outdir / 'recovery_datasets' / f'iteration_{i:04d}_dataset'
            trial_indices = None
            max_trials = self.ift_config['results_to_recovery_dataset'].get('max_trials', None)
            if max_trials is not None:
                print(Fore.GREEN + f"Using only {max_trials}/{self.tpi} trials for learning" + Fore.RESET)
                filenames = list_all_planning_results_trials(planning_results_dir)
                trial_indices = [i for (i, _) in filenames][:max_trials]
            r = ResultsToRecoveryDataset(results_dir=planning_results_dir,
                                         outdir=new_dataset_dir,
                                         labeling_params=self.recovery_labeling_params,
                                         verbose=self.verbose,
                                         trial_indices=trial_indices,
                                         **self.ift_config['results_to_recovery_dataset'])
            r.run()
            new_dataset_dir_rel = new_dataset_dir.relative_to(self.outdir)
            dataset_chunker.store_result('new_dataset_dir', new_dataset_dir_rel.as_posix())
        else:
            new_dataset_dir = self.outdir / new_dataset_dir
        return new_dataset_dir

    def fine_tune(self, iteration_data: IterationData):
        classifier_checkpoint_dir = self.fine_tune_classifier(iteration_data)
        if self.ift_config['fine_tune_recovery'] is None:
            recovery_checkpoint_dir = iteration_data.latest_recovery_checkpoint_dir
        else:
            recovery_checkpoint_dir = self.fine_tune_recovery(iteration_data)
        return classifier_checkpoint_dir, recovery_checkpoint_dir

    def fine_tune_classifier(self, iteration_data: IterationData):
        i = iteration_data.iteration
        if self.ift_config.get('full_retrain_classifier', False):
            latest_checkpoint = pathify(
                self.job_chunker.get_result('initial_classifier_checkpoint')) / 'best_checkpoint'
        else:
            latest_checkpoint = iteration_data.latest_classifier_checkpoint_dir / self.checkpoint_suffix
        fine_tune_chunker = iteration_data.iteration_chunker.sub_chunker('fine tune classifier')
        new_latest_checkpoint_dir = pathify(fine_tune_chunker.get_result('new_latest_checkpoint_dir'))
        if new_latest_checkpoint_dir is None:
            [p.suspend() for p in self.gazebo_processes]

            adaptive_batch_size = compute_batch_size(iteration_data.fine_tuning_classifier_dataset_dirs,
                                                     max_batch_size=16)
            new_latest_checkpoint_dir = fine_tune_classifier(
                train_dataset_dirs=iteration_data.fine_tuning_classifier_dataset_dirs,
                checkpoint=latest_checkpoint,
                log=f'iteration_{i:04d}_classifier_training_logdir',
                trials_directory=self.trials_directory,
                batch_size=adaptive_batch_size,
                verbose=self.verbose,
                validate_first=True,
                model_hparams_update=self.ift_config['labeling_params_update'],
                **self.ift_config['fine_tune_classifier'])
            new_latest_checkpoint_dir_rel = new_latest_checkpoint_dir.relative_to(self.outdir)
            fine_tune_chunker.store_result('new_latest_checkpoint_dir', link_bot_pycommon.pycommon.as_posix())
        else:
            new_latest_checkpoint_dir = self.outdir / new_latest_checkpoint_dir
        return new_latest_checkpoint_dir

    def fine_tune_recovery(self, iteration_data: IterationData):
        i = iteration_data.iteration
        latest_checkpoint = iteration_data.latest_recovery_checkpoint_dir / self.checkpoint_suffix
        fine_tune_chunker = iteration_data.iteration_chunker.sub_chunker('fine tune recovery')
        new_latest_checkpoint_dir = pathify(fine_tune_chunker.get_result('new_latest_checkpoint_dir'))
        if new_latest_checkpoint_dir is None:
            [p.suspend() for p in self.gazebo_processes]

            adaptive_batch_size = compute_batch_size(iteration_data.fine_tuning_recovery_dataset_dirs, max_batch_size=4)
            new_latest_checkpoint_dir = fine_tune_recovery(
                dataset_dirs=iteration_data.fine_tuning_recovery_dataset_dirs,
                checkpoint=latest_checkpoint,
                log=f'iteration_{i:04d}_recovery_training_logdir',
                trials_directory=self.trials_directory,
                batch_size=adaptive_batch_size,
                verbose=self.verbose,
                validate_first=True,
                **self.ift_config['fine_tune_recovery'])
            new_latest_checkpoint_dir_rel = new_latest_checkpoint_dir.relative_to(self.outdir)
            fine_tune_chunker.store_result('new_latest_checkpoint_dir', link_bot_pycommon.pycommon.as_posix())
        else:
            new_latest_checkpoint_dir = self.outdir / new_latest_checkpoint_dir
        return new_latest_checkpoint_dir


@ros_init.with_ros("iterative_fine_tuning")
def main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)
    ou.setLogLevel(ou.LOG_ERROR)
    tf.autograph.set_verbosity(0)

    parser = argparse.ArgumentParser()

    parser.add_argument("outdir", type=pathlib.Path, help='outdir to put results in')
    parser.add_argument("--n-iters", '-n', type=int, help='number of iterations of fine tuning', default=100)
    parser.add_argument("--on-exception", choices=['raise', 'catch', 'retry'], default='retry')

    args = parser.parse_args()

    ift = IterativeFineTuning(args.outdir, on_exception=args.on_exception)
    ift.run(n_iters=args.n_iters)


if __name__ == '__main__':
    main()
