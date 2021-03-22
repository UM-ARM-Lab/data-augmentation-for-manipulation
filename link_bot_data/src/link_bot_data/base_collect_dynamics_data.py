#!/usr/bin/env python
import multiprocessing
import pathlib
from multiprocessing import Process
from time import perf_counter
from typing import Dict, Optional

import numpy as np
from colorama import Fore

import rospy
from link_bot_data.dataset_utils import data_directory, tf_write_example
from link_bot_data.files_dataset import FilesDataset
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.grid_utils import extent_to_env_shape
from link_bot_pycommon.serialization import my_hdump


class BaseDataCollector:

    def __init__(self,
                 scenario_name: str,
                 service_provider: BaseServices,
                 params: Dict,
                 seed: Optional[int] = None,
                 verbose: int = 0):
        self.service_provider = service_provider
        self.params = params
        self.verbose = verbose
        self.scenario_name = scenario_name
        self.scenario = get_scenario(scenario_name)

        if seed is None:
            self.seed = np.random.randint(0, 100)
        else:
            self.seed = seed
        print(Fore.CYAN + f"Using seed: {self.seed}" + Fore.RESET)

        service_provider.setup_env(verbose=self.verbose,
                                   real_time_rate=self.params['real_time_rate'],
                                   max_step_size=self.params['max_step_size'])

    def collect_trajectory(self,
                           traj_idx: int,
                           verbose: int,
                           action_rng: np.random.RandomState,
                           ):
        if self.params['no_objects']:
            rospy.logwarn("Not collecting the environment", logger_name='base_collect_dynamics_data')
            rows, cols, channels = extent_to_env_shape(self.params['extent'], self.params['res'])
            origin = np.array([rows // 2, cols // 2, channels // 2], dtype=np.int32)
            env = np.zeros([rows, cols, channels], dtype=np.float32)
            environment = {'env': env, 'res': self.params['res'], 'origin': origin, 'extent': self.params['extent']}
        else:
            # we assume environment does not change during an individual trajectory
            environment = self.scenario.get_environment(self.params)

        example = {}
        example.update(environment)
        example['traj_idx'] = np.float32(traj_idx)

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

            # # DEBUG
            # grippers_unchanged = np.allclose(state['left_gripper'], last_state['left_gripper'])
            # image_unchanged = np.allclose(state['rgbd'][:, :, :3], last_state['rgbd'][:, :, :3])
            # if image_unchanged and not grippers_unchanged:
            #     rospy.logerr("previous RGB is the same!!!!")
            # last_state = state
            # # END DEBUG

            # TODO: sample the entire action sequence in advance?
            action, invalid = self.scenario.sample_action(action_rng=action_rng,
                                                          environment=environment,
                                                          state=state,
                                                          action_params=self.params,
                                                          validate=True)

            # Visualization
            self.scenario.plot_environment_rviz(environment)
            self.scenario.plot_traj_idx_rviz(traj_idx)
            self.scenario.plot_state_rviz(state, label='actual')
            if time_idx < self.params['steps_per_traj'] - 1:  # skip the last action in visualization as well
                self.scenario.plot_action_rviz(state, action)
            self.scenario.plot_time_idx_rviz(time_idx)
            # End Visualization

            # execute action
            self.scenario.execute_action(action)

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

        return example

    def collect_data(self,
                     n_trajs: int,
                     nickname: str,
                     robot_namespace: str,
                     ):
        outdir = pathlib.Path('fwd_model_data') / nickname
        full_output_directory = data_directory(outdir, n_trajs)

        files_dataset = FilesDataset(full_output_directory)

        full_output_directory.mkdir(exist_ok=True)
        print(Fore.GREEN + full_output_directory.as_posix() + Fore.RESET)

        self.scenario.randomization_initialization(self.params)
        self.scenario.on_before_data_collection(self.params)

        self.save_hparams(full_output_directory, n_trajs, nickname, robot_namespace)

        trial_start = perf_counter()

        combined_seeds = [traj_idx + 100000 * self.seed for traj_idx in range(n_trajs)]
        write_process = None
        queue = multiprocessing.Queue()
        for traj_idx, seed in enumerate(combined_seeds):
            # combine the trajectory idx and the overall "seed" to make a unique seed for each trajectory/seed pair
            env_rng = np.random.RandomState(seed)
            action_rng = np.random.RandomState(seed)

            # Randomize the environment
            randomize = self.params["randomize_n"] and traj_idx % self.params["randomize_n"] == 0
            state = self.scenario.get_state()
            needs_reset = self.scenario.needs_reset(state, self.params)
            if (not self.params['no_objects'] and randomize) or needs_reset:
                if needs_reset:
                    rospy.logwarn("Reset required!")
                self.scenario.randomize_environment(env_rng, self.params)

            # Generate a new trajectory
            example = self.collect_trajectory(traj_idx=traj_idx, verbose=self.verbose, action_rng=action_rng)
            print(f'traj {traj_idx}/{n_trajs} ({seed}), {perf_counter() - trial_start:.4f}s')

            # Save the data
            def _write(_queue):
                full_filename = self.write_example(full_output_directory, example, traj_idx)
                _queue.put(full_filename)

            # we may need to wait before writing again, because this won't parallelize well
            if write_process is not None:
                write_process.join()
            write_process = multiprocessing.Process(target=_write, args=(queue,))
            write_process.start()

        self.scenario.on_after_data_collection(self.params)

        print(Fore.GREEN + full_output_directory.as_posix() + Fore.RESET)

        self.service_provider.pause()

        while not queue.empty():
            outfilename = queue.get()
            files_dataset.add(outfilename)
        return files_dataset

    def save_hparams(self, full_output_directory, n_trajs, nickname, robot_namespace):
        s_for_size = self.scenario.get_state()
        a_for_size, _ = self.scenario.sample_action(action_rng=np.random.RandomState(0),
                                                    environment={},
                                                    state=s_for_size,
                                                    action_params=self.params,
                                                    validate=False)
        state_description = {k: v.shape[0] for k, v in s_for_size.items()}
        action_description = {k: v.shape[0] for k, v in a_for_size.items()}
        dataset_hparams = {
            'nickname':               nickname,
            'robot_namespace':        robot_namespace,
            'seed':                   self.seed,
            'n_trajs':                n_trajs,
            'data_collection_params': self.params,
            'scenario':               self.scenario_name,
            # FIXME: rename this key?
            'scenario_metadata':      self.scenario.dynamics_dataset_metadata(),
            'state_description':      state_description,
            'action_description':     action_description,
        }
        with (full_output_directory / 'hparams.hjson').open('w') as dataset_hparams_file:
            my_hdump(dataset_hparams, dataset_hparams_file, indent=2)

    def write_example(self, full_output_directory, example, traj_idx):
        raise NotImplementedError()


class TfDataCollector(BaseDataCollector):

    def __init__(self,
                 scenario_name: str,
                 service_provider: BaseServices,
                 params: Dict,
                 seed: Optional[int] = None,
                 verbose: int = 0):
        super().__init__(scenario_name=scenario_name,
                         service_provider=service_provider,
                         params=params,
                         seed=seed,
                         verbose=verbose)

    def write_example(self, full_output_directory, example, traj_idx):
        return tf_write_example(example=example, full_output_directory=full_output_directory, example_idx=traj_idx)


class H5DataCollector(BaseDataCollector):

    def __init__(self,
                 scenario_name: str,
                 service_provider: BaseServices,
                 params: Dict,
                 seed: Optional[int] = None,
                 verbose: int = 0):
        super().__init__(scenario_name=scenario_name,
                         service_provider=service_provider,
                         params=params,
                         seed=seed,
                         verbose=verbose)

    def write_example(self, full_output_directory, example, traj_idx):
        # implement this --> h5_write_example(full_output_directory, example, traj_idx)
        pass
