#!/usr/bin/env python
import pathlib
from typing import Dict, Optional

import hjson
import numpy as np
from colorama import Fore
from tqdm import tqdm

import rospy
from arm_robots.robot import RobotPlanningError
from link_bot_data.dataset_utils import make_unique_outdir, tf_write_example, pkl_write_example
from link_bot_data.split_dataset import split_dataset_via_files
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.get_service_provider import get_service_provider
from link_bot_pycommon.serialization import my_hdump


class BaseDataCollector:

    def __init__(self,
                 params: Dict,
                 seed: Optional[int] = None,
                 verbose: int = 0):
        self.params = params
        self.verbose = verbose
        self.scenario_name = self.params['scenario']
        self.scenario = get_scenario(self.scenario_name)
        self.service_provider = get_service_provider(self.params['service_provider'])

        if seed is None:
            self.seed = np.random.randint(0, 100)
        else:
            self.seed = seed
        print(Fore.CYAN + f"Using seed: {self.seed}" + Fore.RESET)

        self.service_provider.setup_env(verbose=self.verbose,
                                        real_time_rate=self.params['real_time_rate'],
                                        max_step_size=self.params['max_step_size'])

    def collect_trajectory(self, traj_idx: int, verbose: int, action_rng: np.random.RandomState):
        environment = self.scenario.get_environment(self.params)

        example = {}
        example.update(environment)
        example['traj_idx'] = np.float32(traj_idx)

        if self.verbose > 0:
            self.scenario.plot_environment_rviz(environment)

        actions = {k: [] for k in self.params['action_keys']}
        # NOTE: state metadata is information that is constant, possibly non-numeric, and convenient to have with state
        #  in most cases it could be considered part of the environment, but sometimes having it with state is better
        states = {k: [] for k in self.params['state_keys'] + self.params['state_metadata_keys']}
        time_indices = []
        last_state = self.scenario.get_state()  # for debugging
        for time_idx in range(self.params['steps_per_traj']):
            # get current state and sample action
            state = self.scenario.get_state()

            # TODO: sample the entire action sequence in advance?
            action, invalid = self.scenario.sample_action(action_rng=action_rng,
                                                          environment=environment,
                                                          state=state,
                                                          action_params=self.params,
                                                          validate=True)
            if invalid:
                rospy.logwarn("unable to sample valid action")
                return {}, invalid

            # Visualization
            if self.verbose > 0:
                self.scenario.plot_environment_rviz(environment)
                self.scenario.plot_traj_idx_rviz(traj_idx)
                self.scenario.plot_state_rviz(state, label='actual')
                if time_idx < self.params['steps_per_traj'] - 1:  # skip the last action in visualization as well
                    self.scenario.plot_action_rviz(state, action)
                self.scenario.plot_time_idx_rviz(time_idx)
            # End Visualization

            # execute action
            try:
                self.scenario.execute_action(environment, state, action)
            except RobotPlanningError:
                rospy.logwarn(f"error executing action {action}")
                return {}, (invalid := True)

            # add to the dataset
            if time_idx < self.params['steps_per_traj'] - 1:  # skip the last action
                for action_name in self.params['action_keys']:
                    action_component = action[action_name]
                    actions[action_name].append(action_component)
            for state_component_name in self.params['state_keys'] + self.params['state_metadata_keys']:
                state_component = state[state_component_name]
                states[state_component_name].append(state_component)
            time_indices.append(time_idx)

        example.update(states)
        example.update(actions)
        example['time_idx'] = np.array(time_indices).astype(np.float32)

        if verbose:
            print(Fore.GREEN + "Trajectory {} Complete".format(traj_idx) + Fore.RESET)

        return example, (invalid := False)

    def collect_data(self,
                     n_trajs: int,
                     nickname: str,
                     root: Optional[pathlib.Path] = None,
                     ):
        if root is None:
            outdir = pathlib.Path('fwd_model_data') / nickname
        else:
            outdir = root / nickname
        full_output_directory = make_unique_outdir(outdir, n_trajs)

        full_output_directory.mkdir(exist_ok=True)
        print(Fore.GREEN + full_output_directory.as_posix() + Fore.RESET)

        self.scenario.randomization_initialization(self.params)
        self.scenario.on_before_data_collection(self.params)
        self.scenario.reset_viz()

        self.save_hparams(full_output_directory, n_trajs, nickname)

        combined_seeds = [traj_idx + 100000 * self.seed for traj_idx in range(n_trajs)]
        for traj_idx, seed in enumerate(tqdm(combined_seeds)):
            invalid = False
            for retry_idx in range(10):
                # combine the trajectory idx and the overall "seed" to make a unique seed for each trajectory/seed pair
                env_rng = np.random.RandomState(seed)
                action_rng = np.random.RandomState(seed)

                # Randomize the environment
                randomize = self.params["randomize_n"] and traj_idx % self.params["randomize_n"] == 0
                state = self.scenario.get_state()
                needs_reset = self.scenario.needs_reset(state, self.params)
                if randomize or needs_reset:
                    if needs_reset:
                        rospy.logwarn("Reset required!")
                    self.scenario.randomize_environment(env_rng, self.params)

                # Generate a new trajectory
                example, invalid = self.collect_trajectory(traj_idx=traj_idx, verbose=self.verbose,
                                                           action_rng=action_rng)
                if not invalid:
                    break

            if invalid:
                raise RuntimeError(f"Could not execute trajectory {traj_idx}")

            # Save the data
            self.write_example(full_output_directory, example, traj_idx)

        self.scenario.on_after_data_collection(self.params)

        print(Fore.GREEN + full_output_directory.as_posix() + Fore.RESET)

        self.service_provider.pause()

        return full_output_directory

    def save_hparams(self, full_output_directory, n_trajs, nickname):
        dataset_hparams = {
            'nickname':               nickname,
            'robot_namespace':        self.scenario.robot.robot_namespace,
            'seed':                   self.seed,
            'n_trajs':                n_trajs,
            'data_collection_params': self.params,
            'scenario':               self.scenario_name,
        }
        with (full_output_directory / 'hparams.hjson').open('w') as dataset_hparams_file:
            my_hdump(dataset_hparams, dataset_hparams_file, indent=2)

    def write_example(self, full_output_directory, example, traj_idx):
        raise NotImplementedError()


class TfDataCollector(BaseDataCollector):

    def __init__(self,
                 params: Dict,
                 seed: Optional[int] = None,
                 verbose: int = 0):
        super().__init__(params=params,
                         seed=seed,
                         verbose=verbose)

    def write_example(self, full_output_directory, example, traj_idx):
        return tf_write_example(example=example, full_output_directory=full_output_directory, example_idx=traj_idx)


class H5DataCollector(BaseDataCollector):

    def __init__(self,
                 params: Dict,
                 seed: Optional[int] = None,
                 verbose: int = 0):
        super().__init__(params=params,
                         seed=seed,
                         verbose=verbose)

    def write_example(self, full_output_directory, example, traj_idx):
        # implement this --> h5_write_example(full_output_directory, example, traj_idx)
        pass


class PklDataCollector(BaseDataCollector):

    def __init__(self,
                 params: Dict,
                 seed: Optional[int] = None,
                 verbose: int = 0):
        super().__init__(params=params,
                         seed=seed,
                         verbose=verbose)

    def write_example(self, full_output_directory, example, traj_idx):
        pkl_write_example(full_output_directory, example, traj_idx)


def collect_dynamics_data(collect_dynamics_params: pathlib.Path,
                          n_trajs: int,
                          nickname: str,
                          verbose=0,
                          save_format: str = 'pkl',
                          seed: Optional[int] = None):
    with collect_dynamics_params.open("r") as f:
        collect_dynamics_params = hjson.load(f)
    DataCollectorClass, extension = get_data_collector_class(save_format)
    data_collector = DataCollectorClass(params=collect_dynamics_params,
                                        seed=seed,
                                        verbose=verbose)
    dataset_dir = data_collector.collect_data(n_trajs=n_trajs, nickname=nickname)
    split_dataset_via_files(dataset_dir, extension)
    return dataset_dir


def get_data_collector_class(save_format: str):
    if save_format == 'h5':
        return H5DataCollector, 'h5'
    elif save_format == 'tfrecord':
        return TfDataCollector, 'tfrecords'
    elif save_format == 'pkl':
        return PklDataCollector, 'pkl'
    else:
        raise NotImplementedError(f"unsupported save_format {save_format}")
