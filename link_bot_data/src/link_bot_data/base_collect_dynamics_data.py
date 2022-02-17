#!/usr/bin/env python
import pathlib
from typing import Dict, Optional

import hjson
import numpy as np
from colorama import Fore
from tqdm import tqdm

import rospy
from arm_robots.robot import RobotPlanningError
from link_bot_data.dataset_utils import make_unique_outdir
from link_bot_data.tf_dataset_utils import tf_write_example, pkl_write_example
from link_bot_data.split_dataset import split_dataset_via_files
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.get_service_provider import get_service_provider
from link_bot_pycommon.serialization import my_hdump
from moonshine.filepath_tools import load_hjson


def idx_seed_gen(base_seed):
    traj_idx = 0
    while True:
        # combine the trajectory idx and self.seed to make a unique seed for each trajectory/seed pair
        seed = traj_idx + 1000 * base_seed
        yield seed

        traj_idx += 1


def collect_trajectory(params,
                       scenario,
                       traj_idx: int,
                       predetermined_start_state,
                       predetermined_actions,
                       verbose: int,
                       action_rng: np.random.RandomState):
    environment = scenario.get_environment(params)

    example = {}
    example.update(environment)
    example['traj_idx'] = np.float32(traj_idx)

    if verbose > 0:
        scenario.plot_environment_rviz(environment)

    actions = {k: [] for k in params['action_keys']}
    # NOTE: state metadata is information that is constant, possibly non-numeric, and convenient to have with state
    #  in most cases it could be considered part of the environment, but sometimes having it with state is better
    states = {k: [] for k in params['state_keys'] + params['state_metadata_keys']}
    time_indices = []

    scenario.clear_action_sampling_state()

    if predetermined_start_state is not None:
        scenario.set_state_from_dict(predetermined_start_state)

    if predetermined_actions is not None:
        n_steps = len(predetermined_actions) + 1
    else:
        n_steps = params['steps_per_traj']

    for time_idx in range(n_steps):
        # get current state and sample action
        state = scenario.get_state()

        # TODO: sample the entire action sequence in advance?
        if predetermined_actions is not None:
            if time_idx < n_steps - 1:
                action = predetermined_actions[time_idx]
            else:
                action = predetermined_actions[-1]
            invalid = False
        else:
            action, invalid = scenario.sample_action(action_rng=action_rng,
                                                     environment=environment,
                                                     state=state,
                                                     action_params=params,
                                                     validate=True)
        if invalid:
            return {}, invalid

        # Visualization
        if verbose > 0:
            scenario.plot_environment_rviz(environment)
            scenario.plot_traj_idx_rviz(traj_idx)
            scenario.plot_state_rviz(state, label='actual')
            if time_idx < n_steps - 1:  # skip the last action in visualization as well
                scenario.plot_action_rviz(state, action)
            scenario.plot_time_idx_rviz(time_idx)
        # End Visualization

        # execute action
        try:
            invalid = scenario.execute_action(environment, state, action)
            if invalid:
                rospy.logwarn(f"error executing action {action}")
                scenario.last_action = None  # to avoid repeating the action that failed
                return {}, (invalid := True)
        except RobotPlanningError:
            rospy.logwarn(f"error executing action {action}")
            scenario.last_action = None  # to avoid repeating the action that failed
            return {}, (invalid := True)

        # add to the dataset
        if time_idx < params['steps_per_traj'] - 1:  # skip the last action
            for action_name in params['action_keys']:
                action_component = action[action_name]
                actions[action_name].append(action_component)
        for state_component_name in params['state_keys'] + params['state_metadata_keys']:
            state_component = state[state_component_name]
            states[state_component_name].append(state_component)
        time_indices.append(time_idx)

    example.update(states)
    example.update(actions)
    example['time_idx'] = np.array(time_indices).astype(np.float32)

    if verbose:
        print(Fore.GREEN + "Trajectory {} Complete".format(traj_idx) + Fore.RESET)

    return example, (invalid := False)


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

    def collect_trajectory(self,
                           traj_idx: int,
                           predetermined_start_state,
                           predetermined_actions,
                           verbose: int,
                           action_rng: np.random.RandomState):
        return collect_trajectory(self.params,
                                  self.scenario,
                                  traj_idx,
                                  predetermined_start_state,
                                  predetermined_actions,
                                  verbose,
                                  action_rng)

    def collect_data(self,
                     n_trajs: int,
                     nickname: str,
                     states_and_actions: Optional = None,
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

        predetermined_start_state = None
        predetermined_actions = None
        if states_and_actions is not None:
            states_and_actions = load_hjson(states_and_actions)

        traj_idx = 0
        for seed in tqdm(idx_seed_gen(self.seed), total=n_trajs):
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

            if states_and_actions is not None:
                predetermined_start_state, predetermined_actions = states_and_actions[traj_idx]

            # Generate a new trajectory
            example, invalid = self.collect_trajectory(traj_idx=traj_idx,
                                                       predetermined_start_state=predetermined_start_state,
                                                       predetermined_actions=predetermined_actions,
                                                       verbose=self.verbose,
                                                       action_rng=action_rng)
            example['seed'] = seed

            if invalid:
                rospy.logwarn(f"Could not execute valid trajectory")
                continue

            # Save the data
            self.write_example(full_output_directory, example, traj_idx)

            # tell the caller that we've made progress
            yield None, traj_idx + 1

            traj_idx += 1

            if traj_idx == n_trajs:
                break

        self.scenario.on_after_data_collection(self.params)

        print(Fore.GREEN + full_output_directory.as_posix() + Fore.RESET)

        self.service_provider.pause()

        yield full_output_directory, n_trajs

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
                          seed: Optional[int] = None,
                          states_and_actions: Optional = None,
                          **kwargs):
    with collect_dynamics_params.open("r") as f:
        collect_dynamics_params = hjson.load(f)
    DataCollectorClass, extension = get_data_collector_class(save_format)
    data_collector = DataCollectorClass(params=collect_dynamics_params,

                                        seed=seed,
                                        verbose=verbose)

    dataset_dir = None
    n_trajs_done = None
    for dataset_dir, n_trajs_done in data_collector.collect_data(n_trajs=n_trajs,
                                                                 states_and_actions=states_and_actions,
                                                                 nickname=nickname):
        if dataset_dir is not None:
            break
        else:
            yield dataset_dir, n_trajs_done

    split_dataset_via_files(dataset_dir, extension)
    yield dataset_dir, n_trajs_done


def get_data_collector_class(save_format: str):
    if save_format == 'h5':
        return H5DataCollector, 'h5'
    elif save_format == 'tfrecord':
        return TfDataCollector, 'tfrecords'
    elif save_format == 'pkl':
        return PklDataCollector, 'pkl'
    else:
        raise NotImplementedError(f"unsupported save_format {save_format}")
