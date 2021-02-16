#!/usr/bin/env python
import argparse
import pathlib
import re
import tempfile
from typing import Dict, List, Optional

import colorama
import numpy as np

from arc_utilities import ros_init
from link_bot_data.classifier_dataset_utils import add_perception_reliability, add_model_error
from link_bot_data.dataset_utils import tf_write_example, add_predicted
from link_bot_gazebo_python.gazebo_services import GazeboServices
from link_bot_planning import results_utils
from link_bot_planning.my_planner import PlanningResult, PlanningQuery, LoggingTree, SetupInfo
from link_bot_planning.test_scenes import get_states_to_save, save_test_scene_given_name
from link_bot_pycommon.args import my_formatter, int_set_arg
from link_bot_pycommon.marker_index_generator import marker_index_generator
from link_bot_pycommon.pycommon import make_dict_tf_float32
from moonshine.filepath_tools import load_json
from moonshine.moonshine_utils import add_batch_single, sequence_of_dicts_to_dict_of_tensors, add_batch, remove_batch


@ros_init.with_ros("results_to_dataset")
def main():
    colorama.init(autoreset=True)
    np.set_printoptions(linewidth=250, precision=3, suppress=True)
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("results_dir", type=pathlib.Path, help='directory containing metrics.json')
    parser.add_argument("labeling_params", type=pathlib.Path, help='labeling params')
    parser.add_argument("outdir", type=pathlib.Path, help='output directory')
    parser.add_argument("--visualize", action='store_true', help='visualize')
    parser.add_argument("--trial-indices", type=int_set_arg, help='which plan(s) to show')
    parser.add_argument("--verbose", '-v', action="count", default=0)

    args = parser.parse_args()

    r = ResultsToDynamicsDataset(args.results_dir,
                                 args.outdir,
                                 args.labeling_params,
                                 args.trial_indices,
                                 args.visualize)


def get_start_idx(outdir: pathlib.Path):
    existing_records = list(outdir.glob(".tfrecords"))
    if len(existing_records) == 0:
        return 0

    start_idx = 0
    for existing_record in existing_records:
        m = re.fullmatch(r'.*?example_([0-9]+)\.tfrecords', existing_record.as_posix())
        record_idx = int(m.group(1))
        start_idx = max(start_idx, record_idx)

    return start_idx


class ResultsToDynamicsDataset:

    def __init__(self,
                 results_dir: pathlib.Path,
                 outdir: pathlib.Path,
                 labeling_params: pathlib.Path,
                 trial_indices: List[int],
                 visualize: bool):
        self.visualize = visualize
        self.viz_id = 0
        self.scenario, self.metadata = results_utils.get_scenario_and_metadata(results_dir)
        self.scenario.on_before_get_state_or_execute_action()
        self.scenario.grasp_rope_endpoints(settling_time=0.0)
        self.service_provider = GazeboServices()

        outdir.mkdir(exist_ok=True, parents=True)

        self.clear_markers()
        self.before_state_idx = marker_index_generator(0)
        self.before_state_pred_idx = marker_index_generator(1)
        self.after_state_idx = marker_index_generator(3)
        self.after_state_pred_idx = marker_index_generator(4)
        self.action_idx = marker_index_generator(5)

        self.labeling_params = load_json(labeling_params)
        self.threshold = self.labeling_params['threshold']

        results_utils.save_dynamics_dataset_hparams(self.scenario, results_dir, outdir, self.metadata)
        self.example_idx = get_start_idx(outdir)

        from time import perf_counter
        t0 = perf_counter()

        for trial_idx, datum in results_utils.trials_generator(results_dir, trial_indices):
            print(f"trial {trial_idx}")
            for example in self.result_datum_to_dynamics_dataset(datum):
                now = perf_counter()
                print(f'{self.example_idx} dt={now - t0:.3f}')
                example.pop('joint_names')
                example = make_dict_tf_float32(example)
                tf_write_example(outdir, example, self.example_idx)
                self.example_idx += 1
                t0 = now

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
        yield from self.dfs(planner_params,
                            planning_query,
                            planning_result.tree,
                            bagfile_name=setup_info.bagfile_name)

    def dfs(self,
            planner_params: Dict,
            planning_query: PlanningQuery,
            tree: LoggingTree,
            bagfile_name: Optional[pathlib.Path] = None,
            depth: int = 0):

        if bagfile_name is None:
            bagfile_name = store_bagfile()

        for child in tree.children:
            # if we only have one child we can skip the restore, this speeds things up a lot
            if len(tree.children) > 1 or depth == 0:
                self.scenario.restore_from_bag_rushed(service_provider=self.service_provider,
                                                      params=planner_params,
                                                      bagfile_name=bagfile_name)

            action = child.action
            before_state, after_state = self.execute(action)

            before_state_pred = {k: v for k, v in tree.state.items()}
            after_state_pred = {k: v for k, v in child.state.items()}

            if self.visualize:
                self.visualize_example(action=action,
                                       after_state=after_state,
                                       before_state=before_state,
                                       before_state_predicted={add_predicted(k): v for k, v in
                                                               before_state_pred.items()},
                                       after_state_predicted={add_predicted(k): v for k, v in after_state_pred.items()},
                                       planning_query=planning_query)

            classifier_horizon = 2  # this script only handles this case
            classifier_start_t = depth

            example_states = sequence_of_dicts_to_dict_of_tensors([before_state, after_state])
            example_states_pred = sequence_of_dicts_to_dict_of_tensors([before_state_pred, after_state_pred])
            example_states_pred.pop("joint_names")
            example_states_pred.pop("num_diverged")
            example_actions = add_batch_single(action)

            example = {
                'classifier_start_t': classifier_start_t,
                'classifier_end_t':   classifier_start_t + classifier_horizon,
                'prediction_start_t': 0,
                'traj_idx':           self.example_idx,
                'time_idx':           [0, 1],
            }
            example.update(planning_query.environment)
            example.update(example_states)
            example.update({add_predicted(k): v for k, v in example_states_pred.items()})
            example.update(example_actions)

            example_batched = add_batch(example)
            actual_batched = add_batch(example_states)
            predicted_batched = add_batch(example_states_pred)

            add_perception_reliability(scenario=self.scenario,
                                       actual=actual_batched,
                                       predictions=predicted_batched,
                                       out_example=example_batched,
                                       labeling_params=self.labeling_params)

            valid_out_examples_batched = add_model_error(self.scenario,
                                                         actual=actual_batched,
                                                         predictions=predicted_batched,
                                                         out_example=example_batched,
                                                         labeling_params=self.labeling_params,
                                                         batch_size=1)
            if valid_out_examples_batched['time_idx'].shape[0] == 1:
                valid_out_example = remove_batch(valid_out_examples_batched)
                yield valid_out_example
            if valid_out_examples_batched['time_idx'].shape[0] > 1:
                raise ValueError()

            yield from self.dfs(planner_params, planning_query, child, depth=depth + 1)

    def execute(self, action: Dict):
        before_state = self.scenario.get_state()
        self.scenario.execute_action(action)
        after_state = self.scenario.get_state()
        return before_state, after_state

    def visualize_example(self,
                          action: Dict,
                          after_state: Dict,
                          before_state: Dict,
                          after_state_predicted: Dict,
                          before_state_predicted: Dict,
                          planning_query: PlanningQuery):
        self.scenario.plot_environment_rviz(planning_query.environment)
        self.scenario.plot_state_rviz(before_state, idx=next(self.before_state_idx), label='actual')
        self.scenario.plot_state_rviz(before_state_predicted, idx=next(self.before_state_pred_idx), label='predicted')
        self.scenario.plot_state_rviz(after_state, idx=next(self.after_state_idx), label='actual')
        self.scenario.plot_state_rviz(after_state_predicted, idx=next(self.after_state_pred_idx), label='predicted')
        self.scenario.plot_action_rviz(before_state, action, idx=next(self.action_idx), label='actual')
        self.viz_id += 1

    def clear_markers(self):
        self.scenario.reset_planning_viz()


def store_bagfile():
    joint_state, links_states = get_states_to_save()
    bagfile_name = pathlib.Path(tempfile.NamedTemporaryFile().name)
    return save_test_scene_given_name(joint_state, links_states, bagfile_name)


if __name__ == '__main__':
    main()
