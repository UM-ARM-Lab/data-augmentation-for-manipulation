#!/usr/bin/env python
import argparse
import pathlib
import tempfile
from time import perf_counter, sleep
from typing import Dict, List, Optional

import colorama
import numpy as np

import rospy
from arc_utilities import ros_init
from link_bot_data.classifier_dataset_utils import add_perception_reliability, add_model_error
from link_bot_data.dataset_utils import tf_write_example, add_predicted
from link_bot_data.files_dataset import FilesDataset
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_planning import results_utils
from link_bot_planning.my_planner import PlanningQuery, LoggingTree
from link_bot_planning.results_utils import get_transitions
from link_bot_planning.test_scenes import get_states_to_save, save_test_scene_given_name
from link_bot_pycommon.args import my_formatter, int_set_arg, BooleanAction
from link_bot_pycommon.job_chunking import JobChunker
from link_bot_pycommon.marker_index_generator import marker_index_generator
from link_bot_pycommon.pycommon import deal_with_exceptions, try_make_dict_tf_float32
from moonshine.filepath_tools import load_hjson
from moonshine.moonshine_utils import add_batch_single, sequence_of_dicts_to_dict_of_tensors, add_batch, remove_batch
from std_msgs.msg import Empty


@ros_init.with_ros("results_to_dataset")
def main():
    colorama.init(autoreset=True)
    np.set_printoptions(linewidth=250, precision=3, suppress=True)
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("results_dir", type=pathlib.Path, help='directory containing metrics.json')
    parser.add_argument("outdir", type=pathlib.Path, help='output directory')
    parser.add_argument('--full-tree', action=BooleanAction, required=True)
    parser.add_argument("--labeling-params", type=pathlib.Path, help='labeling params',
                        default=pathlib.Path('labeling_params/classifier/dual.hjson'))
    parser.add_argument("--visualize", action='store_true', help='visualize')
    parser.add_argument("--gui", action='store_true', help='show gzclient, the gazebo gui')
    parser.add_argument("--launch", type=str, help='launch file name')
    parser.add_argument("--world", type=str, help='world file name')
    parser.add_argument("--trial-indices", type=int_set_arg, help='which plan(s) to show')
    parser.add_argument("--subsample-fraction", type=float, default=1.0, help='number between 0 and 1')

    args = parser.parse_args()

    r = ResultsToClassifierDataset(args.results_dir,
                                   args.outdir,
                                   args.labeling_params,
                                   args.trial_indices,
                                   args.full_tree,
                                   args.visualize,
                                   args.gui,
                                   args.launch,
                                   args.world,
                                   args.subsample_fraction)


def compute_example_idx(trial_idx, example_idx_for_trial):
    return 10_000 * trial_idx + example_idx_for_trial


class ResultsToClassifierDataset:

    def __init__(self,
                 results_dir: pathlib.Path,
                 outdir: pathlib.Path,
                 labeling_params: pathlib.Path,
                 trial_indices: List[int],
                 visualize: bool,
                 full_tree: bool,
                 gui: bool,
                 launch: str,
                 world: str,
                 subsample_fraction: float):
        self.restart = False
        self.rng = np.random.RandomState(0)
        self.service_provider = GazeboServices()
        self.full_tree = full_tree

        self.visualize = visualize
        self.viz_id = 0
        self.scenario, self.metadata = results_utils.get_scenario_and_metadata(results_dir)

        self.files = FilesDataset(outdir)

        if self.full_tree:
            self.scenario.on_before_get_state_or_execute_action()
            self.scenario.grasp_rope_endpoints(settling_time=0.0)

        outdir.mkdir(exist_ok=True, parents=True)

        self.labeling_params = load_hjson(labeling_params)
        self.threshold = self.labeling_params['threshold']

        self.gazebo_restarting_sub = rospy.Subscriber("gazebo_restarting", Empty, self.on_gazebo_restarting)

        results_utils.save_dynamics_dataset_hparams(results_dir, outdir, self.metadata)

        if self.full_tree:
            def _on_exception():
                self.restart = False
                sleep(10)
                self.scenario.on_before_get_state_or_execute_action()
                self.scenario.grasp_rope_endpoints(settling_time=0.0)

            def _results_to_classifier_dataset():
                self.full_results_to_classifier_dataset(results_dir, trial_indices, outdir, subsample_fraction)
        else:
            def _on_exception():
                pass

            def _results_to_classifier_dataset():
                self.results_to_classifier_dataset(results_dir, trial_indices, outdir, subsample_fraction)

        deal_with_exceptions(how_to_handle='raise',
                             function=_results_to_classifier_dataset,
                             exception_callback=_on_exception,
                             print_exception=True,
                             )

    def results_to_classifier_dataset(self,
                                      results_dir: pathlib.Path,
                                      trial_indices,
                                      outdir: pathlib.Path,
                                      subsample_fraction: float):
        logfilename = outdir / 'logfile.hjson'
        job_chunker = JobChunker(logfilename)

        t0 = perf_counter()
        last_t = t0
        total_examples = 0
        for trial_idx, datum in results_utils.trials_generator(results_dir, trial_indices):
            if job_chunker.result_exists(str(trial_idx)):
                rospy.loginfo(f"Found existing classifier data for trial {trial_idx}")
                continue

            self.clear_markers()
            self.before_state_idx = marker_index_generator(0)
            self.before_state_pred_idx = marker_index_generator(1)
            self.after_state_idx = marker_index_generator(3)
            self.after_state_pred_idx = marker_index_generator(4)
            self.action_idx = marker_index_generator(5)

            example_idx_for_trial = 0

            self.example_idx = compute_example_idx(trial_idx, example_idx_for_trial)
            for example in self.result_datum_to_dynamics_dataset(datum, trial_idx, subsample_fraction):
                now = perf_counter()
                dt = now - last_t
                total_dt = now - t0
                last_t = now

                self.example_idx = compute_example_idx(trial_idx, example_idx_for_trial)
                total_examples += 1
                print(
                    f'Trial {trial_idx} Example {self.example_idx} dt={dt:.3f}, total time={total_dt:.3f}, {total_examples=}')
                example = try_make_dict_tf_float32(example)
                full_filename = tf_write_example(outdir, example, self.example_idx)
                self.files.add(full_filename)
                example_idx_for_trial += 1

                job_chunker.store_result(trial_idx, {'trial':              trial_idx,
                                                     'examples for trial': example_idx_for_trial})

            job_chunker.store_result(trial_idx, {'trial':              trial_idx,
                                                 'examples for trial': example_idx_for_trial})

        self.files.split()

    def full_results_to_classifier_dataset(self,
                                           results_dir: pathlib.Path,
                                           trial_indices,
                                           outdir: pathlib.Path,
                                           subsample_fraction: float):
        logfilename = outdir / 'logfile.hjson'
        job_chunker = JobChunker(logfilename)

        t0 = perf_counter()
        last_t = t0
        max_examples_per_trial = 500
        enough_trials_msg = f"moving on to next trial, already got {max_examples_per_trial} examples from this trial"
        total_examples = 0
        for trial_idx, datum in results_utils.trials_generator(results_dir, trial_indices):
            if job_chunker.result_exists(str(trial_idx)):
                rospy.loginfo(f"Found existing classifier data for trial {trial_idx}")
                continue

            example_idx_for_trial = 0

            self.example_idx = compute_example_idx(trial_idx, example_idx_for_trial)
            for example in self.full_result_datum_to_dynamics_dataset(datum, trial_idx, subsample_fraction):
                now = perf_counter()
                dt = now - last_t
                total_dt = now - t0
                last_t = now

                self.example_idx = compute_example_idx(trial_idx, example_idx_for_trial)
                total_examples += 1
                print(
                    f'Trial {trial_idx} Example {self.example_idx} dt={dt:.3f}, total time={total_dt:.3f}, {total_examples=}')
                example = try_make_dict_tf_float32(example)
                tf_write_example(outdir, example, self.example_idx)
                example_idx_for_trial += 1

                if example_idx_for_trial > 50:
                    job_chunker.store_result(trial_idx, {'trial':              trial_idx,
                                                         'examples for trial': example_idx_for_trial})
                if example_idx_for_trial > max_examples_per_trial:
                    rospy.logwarn(enough_trials_msg)
                    break

            job_chunker.store_result(trial_idx, {'trial':              trial_idx,
                                                 'examples for trial': example_idx_for_trial})

    def result_datum_to_dynamics_dataset(self, datum: Dict, trial_idx: int, subsample_fraction: float):
        for t, transition in enumerate(get_transitions(datum)):
            environment, (before_state_pred, before_state), action, (after_state_pred, after_state), _ = transition
            if self.visualize:
                self.visualize_example(action=action,
                                       after_state=after_state,
                                       before_state=before_state,
                                       before_state_predicted={add_predicted(k): v for k, v in
                                                               before_state_pred.items()},
                                       after_state_predicted={add_predicted(k): v for k, v in after_state_pred.items()},
                                       environment=environment)

            yield from self.generate_example(
                environment=environment,
                action=action,
                before_state=before_state,
                before_state_pred=before_state_pred,
                after_state=after_state,
                after_state_pred=after_state_pred,
                classifier_start_t=t,
            )

    def full_result_datum_to_dynamics_dataset(self, datum: Dict, trial_idx: int, subsample_fraction: float):
        steps = datum['steps']
        setup_info = datum['setup_info']
        planner_params = datum['planner_params']
        for step in steps:
            if step['type'] == 'executed_plan':
                planning_result = step['planning_result']
                planning_query = step['planning_query']
                yield from self.dfs(planner_params,
                                    planning_query,
                                    planning_result.tree,
                                    bagfile_name=setup_info.bagfile_name,
                                    subsample_fraction=subsample_fraction)

    def dfs(self,
            planner_params: Dict,
            planning_query: PlanningQuery,
            tree: LoggingTree,
            subsample_fraction: float,
            bagfile_name: Optional[pathlib.Path] = None,
            depth: int = 0,
            ):

        if self.restart:
            raise RuntimeError()

        if bagfile_name is None:
            bagfile_name = store_bagfile()

        for child in tree.children:
            # uniformly randomly sub-sample? this is currently broken
            # r = self.rng.uniform()
            # if r < subsample_fraction:
            skip_restore = len(tree.children) > 1 or depth == 0
            # if we only have one child we can skip the restore, this speeds things up a lot
            if skip_restore:
                deal_with_exceptions('retry',
                                     lambda: self.scenario.restore_from_bag_rushed(
                                         service_provider=self.service_provider,
                                         params=planner_params,
                                         bagfile_name=bagfile_name))
            before_state, after_state = self.execute(environment=planning_query.environment, action=child.action)
            if self.visualize:
                self.visualize_example(action=child.action,
                                       after_state=after_state,
                                       before_state=before_state,
                                       before_state_predicted={add_predicted(k): v for k, v in
                                                               tree.state.items()},
                                       after_state_predicted={add_predicted(k): v for k, v in child.state.items()},
                                       environment=planning_query.environment)

            yield from self.generate_example(
                environment=planning_query.environment,
                action=child.action,
                before_state=before_state,
                before_state_pred=tree.state,
                after_state=after_state,
                after_state_pred=child.state,
                classifier_start_t=depth,
            )
            # recursion
            yield from self.dfs(planner_params, planning_query, child, subsample_fraction, depth=depth + 1)

    def generate_example(self,
                         environment: Dict,
                         action: Dict,
                         before_state: Dict,
                         before_state_pred: Dict,
                         after_state: Dict,
                         after_state_pred: Dict,
                         classifier_start_t: int):
        classifier_horizon = 2  # this script only handles this case
        example_states = sequence_of_dicts_to_dict_of_tensors([before_state, after_state])
        example_states_pred = sequence_of_dicts_to_dict_of_tensors([before_state_pred, after_state_pred])
        if 'num_diverged' in example_states_pred:
            example_states_pred.pop("num_diverged")
        example_actions = add_batch_single(action)
        example = {
            'classifier_start_t': classifier_start_t,
            'classifier_end_t':   classifier_start_t + classifier_horizon,
            'prediction_start_t': 0,
            'traj_idx':           self.example_idx,
            'time_idx':           [0, 1],
        }
        example.update(environment)
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
        test_shape = valid_out_examples_batched['time_idx'].shape[0]
        if test_shape == 1:
            valid_out_example = remove_batch(valid_out_examples_batched)
            yield valid_out_example
        elif test_shape > 1:
            raise ValueError()

    def execute(self, environment: Dict, action: Dict):
        self.service_provider.play()
        before_state = self.scenario.get_state()
        self.scenario.execute_action(environment=environment, state=before_state, action=action)
        after_state = self.scenario.get_state()
        return before_state, after_state

    def visualize_example(self,
                          action: Dict,
                          after_state: Dict,
                          before_state: Dict,
                          after_state_predicted: Dict,
                          before_state_predicted: Dict,
                          environment: Dict):
        self.scenario.plot_environment_rviz(environment)
        self.scenario.plot_state_rviz(before_state, idx=next(self.before_state_idx), label='actual')
        self.scenario.plot_state_rviz(before_state_predicted, idx=next(self.before_state_pred_idx), label='predicted',
                                      color='blue')
        self.scenario.plot_state_rviz(after_state, idx=next(self.after_state_idx), label='actual')
        self.scenario.plot_state_rviz(after_state_predicted, idx=next(self.after_state_pred_idx), label='predicted',
                                      color='blue')
        self.scenario.plot_action_rviz(before_state, action, idx=next(self.action_idx), label='actual')
        self.viz_id += 1

    def clear_markers(self):
        self.scenario.reset_planning_viz()

    def on_gazebo_restarting(self, msg: Empty):
        self.restart = True


def store_bagfile():
    joint_state, links_states = get_states_to_save()
    bagfile_name = pathlib.Path(tempfile.NamedTemporaryFile().name)
    return save_test_scene_given_name(joint_state, links_states, bagfile_name, force=True)


if __name__ == '__main__':
    main()
