#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict, List

import colorama
import numpy as np

from arc_utilities import ros_init
from link_bot_data.dataset_utils import tf_write_example
from link_bot_gazebo_python.gazebo_services import GazeboServices
from link_bot_planning import results_utils
from link_bot_planning.my_planner import PlanningResult, PlanningQuery, LoggingTree, SetupInfo
from link_bot_pycommon.args import my_formatter, int_set_arg
from link_bot_pycommon.pycommon import make_dict_tf_float32
from moonshine.moonshine_utils import add_batch_single, sequence_of_dicts_to_dict_of_tensors


@ros_init.with_ros("results_to_dynamics_dataset")
def main():
    colorama.init(autoreset=True)
    np.set_printoptions(linewidth=250, precision=3, suppress=True)
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("results_dir", type=pathlib.Path, help='directory containing metrics.json')
    parser.add_argument("outdir", type=pathlib.Path, help='output directory')
    parser.add_argument("--trial-indices", type=int_set_arg, help='which plan(s) to show')
    parser.add_argument("--verbose", '-v', action="count", default=0)

    args = parser.parse_args()

    r = ResultsToDynamicsDataset(args.results_dir, args.outdir, args.trial_indices)


class ResultsToDynamicsDataset:

    def __init__(self, results_dir: pathlib.Path, outdir: pathlib.Path, trial_indices: List[int]):
        self.viz_id = 0
        self.scenario, self.metadata = results_utils.get_scenario_and_metadata(results_dir)
        self.scenario.on_before_get_state_or_execute_action()
        self.service_provider = GazeboServices()

        outdir.mkdir(exist_ok=True, parents=True)

        results_utils.save_dynamics_dataset_hparams(self.scenario, results_dir, outdir, self.metadata)
        self.example_idx = 0
        for trial_idx, datum in results_utils.trials_generator(results_dir, trial_indices):
            print(f"trial {trial_idx}")
            for example in self.result_datum_to_dynamics_dataset(datum):
                example.pop('joint_names')
                example = make_dict_tf_float32(example)
                tf_write_example(outdir, example, self.example_idx)
                self.example_idx += 1

    def result_datum_to_dynamics_dataset(self, datum: Dict):
        steps = datum['steps']
        setup_info = datum['setup_info']
        planner_params = datum['planner_params']
        for step in steps:
            if step['type'] == 'executed_plan':
                planning_result = step['planning_result']
                planning_query = step['planning_query']
                yield from self.planning_result_to_dynamics_dataset(planner_params,
                                                                    setup_info,
                                                                    planning_query,
                                                                    planning_result)

    def planning_result_to_dynamics_dataset(self,
                                            planner_params: Dict,
                                            setup_info: SetupInfo,
                                            planning_query: PlanningQuery,
                                            planning_result: PlanningResult):
        planned_states = [planning_query.start]
        yield from self.dfs(planner_params, setup_info, planning_query, planning_result.tree, planned_states, [])

    def dfs(self,
            planner_params: Dict,
            setup_info: SetupInfo,
            planning_query: PlanningQuery,
            tree: LoggingTree,
            planned_states: List[Dict],
            actions: List[Dict]):
        if len(tree.children) == 0:
            start_state = planning_query.start

            self.scenario.restore_from_bag(service_provider=self.service_provider,
                                           params=planner_params,
                                           bagfile_name=setup_info.bagfile_name)

            before_state = start_state
            actual_states = [before_state]
            for action in actions:
                self.scenario.execute_action(action)
                after_state = self.scenario.get_state()
                actual_states.append(after_state)

                self.scenario.plot_environment_rviz(planning_query.environment)
                self.scenario.plot_state_rviz(before_state, idx=2 * self.viz_id + 0)
                self.scenario.plot_action_rviz(before_state, action, idx=2 * self.viz_id + 0)
                self.scenario.plot_state_rviz(after_state, idx=2 * self.viz_id + 1)
                self.viz_id += 1

                example = planning_query.environment
                example['traj_idx'] = [self.example_idx, self.example_idx]
                example_states = sequence_of_dicts_to_dict_of_tensors([before_state, after_state])
                example_actions = add_batch_single(action)
                example.update(example_states)
                example.update(example_actions)
                example['time_idx'] = [0, 1]
                yield example

                before_state = after_state

        for child in tree.children:
            yield from self.dfs(planner_params,
                                setup_info,
                                planning_query,
                                child,
                                planned_states + [child.state],
                                actions + [child.action])


if __name__ == '__main__':
    main()
