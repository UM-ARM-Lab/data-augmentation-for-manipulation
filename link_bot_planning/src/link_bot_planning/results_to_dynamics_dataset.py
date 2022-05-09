import pathlib
from typing import Dict, Optional

import numpy as np
from colorama import Fore
from tqdm import tqdm

import rospy
from analysis import results_utils
from analysis.results_utils import NoTransitionsError
from arc_utilities.algorithms import chunked_iterable
from link_bot_data.dataset_utils import DEFAULT_VAL_SPLIT, DEFAULT_TEST_SPLIT
from link_bot_data.split_dataset import split_dataset
from link_bot_data.tf_dataset_utils import write_example
from link_bot_planning.my_planner import PlanningResult
from link_bot_planning.trial_result import ExecutionResult
from link_bot_pycommon.serialization import my_hdump
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_np_arrays
from moonshine.numpify import numpify


def compute_example_idx(trial_idx, example_idx_for_trial):
    return 10_000 * trial_idx + example_idx_for_trial


class ResultsToDynamicsDataset:

    def __init__(self,
                 results_dir: pathlib.Path,
                 outdir: pathlib.Path,
                 traj_length: Optional[int] = None,
                 val_split=DEFAULT_VAL_SPLIT,
                 test_split=DEFAULT_TEST_SPLIT):
        self.traj_length = traj_length
        self.results_dir = results_dir
        self.trials = (list(results_utils.trials_generator(self.results_dir)))
        self.outdir = outdir
        self.val_split = val_split
        self.test_split = test_split

        self.scenario, self.metadata = results_utils.get_scenario_and_metadata(results_dir)

        self.example_idx = None

        outdir.mkdir(exist_ok=True, parents=True)

    def run(self):
        self.save_hparams()
        self.results_to_classifier_dataset()
        split_dataset(self.outdir, val_split=self.val_split, test_split=self.test_split)

    def save_hparams(self):
        # FIXME: hard-coded
        planner_params = self.metadata['planner_params']

        dataset_hparams = {
            'scenario':               planner_params['scenario'],
            'from_results':           self.results_dir,
            'seed':                   None,
            'n_trajs':                len(self.trials),
            'data_collection_params': {
                'scenario_params':            planner_params.get("scenario_params", {}),
                'service_provider':           'gazebo',
                'state_description':          {
                    'left_gripper':    3,
                    'right_gripper':   3,
                    'joint_positions': 18,
                    'rope':            75,
                },
                'state_metadata_description': {
                    'joint_names': None,
                },
                'action_description':         {
                    'left_gripper_position':  3,
                    'right_gripper_position': 3,
                },
                'env_description':            {
                    'env':          None,
                    'extent':       4,
                    'origin_point': 3,
                    'res':          None,
                    'scene_msg':    None,
                    'sdf':          None,
                    'sdf_grad':     None,
                },
            },
        }
        with (self.outdir / 'hparams.hjson').open('w') as dataset_hparams_file:
            my_hdump(dataset_hparams, dataset_hparams_file, indent=2)

    def results_to_classifier_dataset(self):
        total_examples = 0
        for trial_idx, datum, _ in self.trials:
            self.scenario.heartbeat()

            example_idx_for_trial = 0

            self.example_idx = compute_example_idx(trial_idx, example_idx_for_trial)
            try:
                examples_gen = self.result_datum_to_dynamics_dataset(datum, trial_idx)
                for example in tqdm(examples_gen):
                    self.example_idx = compute_example_idx(trial_idx, example_idx_for_trial)
                    total_examples += 1
                    write_example(self.outdir, example, self.example_idx, 'pkl')
                    example_idx_for_trial += 1
            except NoTransitionsError:
                rospy.logerr(f"Trial {trial_idx} had no transitions")
                pass

        print(Fore.LIGHTMAGENTA_EX + f"Wrote {total_examples} examples" + Fore.RESET)

    def result_datum_to_dynamics_dataset(self, datum: Dict, trial_idx: int):
        steps = datum['steps']

        if len(steps) == 0:
            raise NoTransitionsError()

        actions = []
        states = []
        for step_idx, step in enumerate(steps):
            if step['type'] == 'executed_plan':
                planning_result: PlanningResult = step['planning_result']
                execution_result: ExecutionResult = step['execution_result']
                actions_step = planning_result.actions
                states_step = execution_result.path
            elif step['type'] == 'executed_recovery':
                execution_result: ExecutionResult = step['execution_result']
                recovery_action = step['recovery_action']
                actions_step = [recovery_action]
                states_step = execution_result.path
            else:
                raise NotImplementedError(f"invalid step type {step['type']}")

            if len(actions_step) == 0 or actions_step[0] is None:
                continue

            actions_step = numpify(actions_step)
            states_step = numpify(states_step)

            actions.extend(actions_step)
            states.extend(states_step)

        if self.traj_length is not None:
            action_subsequences = list(reversed(list(chunked_iterable(reversed(actions), self.traj_length))))
            action_subsequences = [a_seq[:-1] for a_seq in action_subsequences]
            state_subsequences = list(reversed(list(chunked_iterable(reversed(states), self.traj_length))))
        else:
            action_subsequences = [actions]
            state_subsequences = [states]

        for action_subsequence, state_subsequence in zip(action_subsequences, state_subsequences):
            actions_dict = sequence_of_dicts_to_dict_of_np_arrays(action_subsequence, 0)
            states_dict = sequence_of_dicts_to_dict_of_np_arrays(state_subsequence, 0)

            time_idx = np.arange(len(state_subsequence))
            environment = steps[0]['planning_query'].environment
            example = {
                'traj_idx': trial_idx,
                'time_idx': time_idx,
            }
            example.update(environment)
            example.update(actions_dict)
            example.update(states_dict)

            example.pop("stdev", None)
            example.pop("error", None)
            example.pop("num_diverged", None)

            yield example
