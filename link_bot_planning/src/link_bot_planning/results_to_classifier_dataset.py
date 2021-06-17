import pathlib
import tempfile
from typing import Optional, List, Dict, Union

import numpy as np
from colorama import Fore
from progressbar import progressbar

import rospy
from arm_robots.robot import RobotPlanningError
from link_bot_data.classifier_dataset_utils import add_perception_reliability, add_model_error_and_filter
from link_bot_data.dataset_utils import add_predicted, write_example
from link_bot_data.split_dataset import split_dataset
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_planning.analysis import results_utils
from link_bot_planning.analysis.results_utils import NoTransitionsError, get_transitions, \
    dynamics_dataset_params_from_classifier_params, classifier_params_from_planner_params, \
    fwd_model_params_from_planner_params
from link_bot_planning.my_planner import PlanningQuery, LoggingTree
from link_bot_planning.test_scenes import get_states_to_save, save_test_scene_given_name
from link_bot_pycommon.job_chunking import JobChunker
from link_bot_pycommon.marker_index_generator import marker_index_generator
from link_bot_pycommon.pycommon import deal_with_exceptions
from link_bot_pycommon.serialization import my_hdump
from moonshine.filepath_tools import load_hjson
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_tensors, add_batch_single, add_batch, remove_batch
from std_msgs.msg import Empty


def compute_example_idx(trial_idx, example_idx_for_trial):
    return 10_000 * trial_idx + example_idx_for_trial


class ResultsToClassifierDataset:

    def __init__(self,
                 results_dir: pathlib.Path,
                 outdir: pathlib.Path,
                 labeling_params: Optional[Union[pathlib.Path, Dict]] = None,
                 trial_indices: Optional[List[int]] = None,
                 visualize: bool = False,
                 save_format: str = 'pkl',
                 full_tree: bool = False,
                 regenerate: bool = False,
                 verbose: int = 1,
                 only_rejected_transitions: bool = False,
                 max_examples_per_trial: Optional[int] = 500,
                 **kwargs):
        self.restart = False
        self.save_format = save_format
        self.rng = np.random.RandomState(0)
        self.service_provider = GazeboServices()
        self.full_tree = full_tree
        self.results_dir = results_dir
        self.outdir = outdir
        self.trial_indices = trial_indices
        self.verbose = verbose
        self.regenerate = regenerate
        self.only_rejected_transitions = only_rejected_transitions
        self.max_examples_per_trial = max_examples_per_trial

        if labeling_params is None:
            labeling_params = pathlib.Path('labeling_params/classifier/dual.hjson')

        if isinstance(labeling_params, Dict):
            self.labeling_params = labeling_params
        else:
            self.labeling_params = load_hjson(labeling_params)

        self.threshold = self.labeling_params['threshold']

        self.visualize = visualize
        self.scenario, self.metadata = results_utils.get_scenario_and_metadata(results_dir)

        self.example_idx = None

        if self.full_tree:
            self.scenario.on_before_get_state_or_execute_action()
            self.scenario.grasp_rope_endpoints(settling_time=0.0)

        outdir.mkdir(exist_ok=True, parents=True)

        self.gazebo_restarting_sub = rospy.Subscriber("gazebo_restarting", Empty, self.on_gazebo_restarting)

        self.before_state_idx = marker_index_generator(0)
        self.before_state_pred_idx = marker_index_generator(1)
        self.after_state_idx = marker_index_generator(3)
        self.after_state_pred_idx = marker_index_generator(4)
        self.action_idx = marker_index_generator(5)

    def run(self):
        self.save_hparams()

        if self.full_tree:
            self.full_results_to_classifier_dataset()
        else:
            self.results_to_classifier_dataset()

        split_dataset(self.outdir)

    def save_hparams(self):
        planner_params = self.metadata['planner_params']
        classifier_params = classifier_params_from_planner_params(planner_params)
        try:
            phase2_dataset_params = dynamics_dataset_params_from_classifier_params(classifier_params)
        except KeyError:
            # this happens when you run the planner without any of the learned classifiers
            fwd_model_hparams = fwd_model_params_from_planner_params(planner_params)
            phase2_dataset_params = {
                'scenario':             planner_params['scenario'],
                'env_keys':             [
                    'env',
                    'origin',
                    'origin_point',
                    'res',
                    'extent',
                    'scene_msg',
                ],
                'true_state_keys':      fwd_model_hparams['state_keys'],
                'predicted_state_keys': fwd_model_hparams['state_keys'],
                'state_metadata_keys':  fwd_model_hparams['state_metadata_keys'],
                'action_keys':          fwd_model_hparams['action_keys'],
                'labeling_params':      self.labeling_params,
            }

        dataset_hparams = phase2_dataset_params
        dataset_hparams_update = {
            'from_results':           self.results_dir,
            'seed':                   None,
            'data_collection_params': {
                'steps_per_traj': 2,
            },
        }
        dataset_hparams.update(dataset_hparams_update)
        with (self.outdir / 'hparams.hjson').open('w') as dataset_hparams_file:
            my_hdump(dataset_hparams, dataset_hparams_file, indent=2)

    def results_to_classifier_dataset(self):
        logfilename = self.outdir / 'logfile.hjson'
        job_chunker = JobChunker(logfilename)

        total_examples = 0
        for trial_idx, datum in results_utils.trials_generator(self.results_dir, self.trial_indices):
            self.scenario.heartbeat()

            if job_chunker.has_result(str(trial_idx)):
                rospy.loginfo(f"Found existing classifier data for trial {trial_idx}")
                continue

            self.reset_visualization()

            example_idx_for_trial = 0

            self.example_idx = compute_example_idx(trial_idx, example_idx_for_trial)
            try:
                examples_gen = self.result_datum_to_dynamics_dataset(datum, trial_idx)
                for example in progressbar(examples_gen):
                    self.example_idx = compute_example_idx(trial_idx, example_idx_for_trial)
                    total_examples += 1
                    write_example(self.outdir, example, self.example_idx, self.save_format)
                    example_idx_for_trial += 1

                    job_chunker.store_result(trial_idx, {'trial':              trial_idx,
                                                         'examples for trial': example_idx_for_trial})
            except NoTransitionsError:
                rospy.logerr(f"Trial {trial_idx} had no transitions")
                pass

            job_chunker.store_result(trial_idx, {'trial':              trial_idx,
                                                 'examples for trial': example_idx_for_trial})

        print(Fore.LIGHTMAGENTA_EX + f"Wrote {total_examples} classifier examples" + Fore.RESET)

    def reset_visualization(self):
        self.clear_markers()
        self.before_state_idx = marker_index_generator(0)
        self.before_state_pred_idx = marker_index_generator(1)
        self.after_state_idx = marker_index_generator(3)
        self.after_state_pred_idx = marker_index_generator(4)
        self.action_idx = marker_index_generator(5)

    def full_results_to_classifier_dataset(self):
        logfilename = self.outdir / 'logfile.hjson'
        job_chunker = JobChunker(logfilename)

        enough_trials_msg = f"moving to next trial, already got {self.max_examples_per_trial} examples from this trial"
        total_examples = 0
        for trial_idx, datum in results_utils.trials_generator(self.results_dir, self.trial_indices):
            if job_chunker.has_result(str(trial_idx)) and not self.regenerate:
                rospy.loginfo(f"Found existing classifier data for trial {trial_idx}")
                continue

            example_idx_for_trial = 0

            self.reset_visualization()

            self.example_idx = compute_example_idx(trial_idx, example_idx_for_trial)
            for example in self.full_result_datum_to_dynamics_dataset(datum, trial_idx):
                self.example_idx = compute_example_idx(trial_idx, example_idx_for_trial)
                total_examples += 1
                write_example(self.outdir, example, self.example_idx, self.save_format)
                example_idx_for_trial += 1

                if example_idx_for_trial > 50:
                    job_chunker.store_result(trial_idx, {'trial':              trial_idx,
                                                         'examples for trial': example_idx_for_trial})
                if self.max_examples_per_trial is not None and example_idx_for_trial > self.max_examples_per_trial:
                    rospy.logwarn(enough_trials_msg)
                    break

            job_chunker.store_result(trial_idx, {'trial':              trial_idx,
                                                 'examples for trial': example_idx_for_trial})

    def result_datum_to_dynamics_dataset(self, datum: Dict, trial_idx: int):
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

    def full_result_datum_to_dynamics_dataset(self, datum: Dict, trial_idx: int):
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
                                    bagfile_name=setup_info.bagfile_name)

    def dfs(self,
            planner_params: Dict,
            planning_query: PlanningQuery,
            tree: LoggingTree,
            bagfile_name: Optional[pathlib.Path] = None,
            depth: int = 0,
            ):

        if self.restart:
            raise RuntimeError()

        if bagfile_name is None:
            bagfile_name = store_bagfile()

        for child in tree.children:
            # if we only have one child we can skip the restore, this speeds things up a lot
            must_restore = len(tree.children) > 1
            if must_restore:
                deal_with_exceptions('retry',
                                     lambda: self.scenario.restore_from_bag_rushed(
                                         service_provider=self.service_provider,
                                         params=planner_params,
                                         bagfile_name=bagfile_name))
            before_state, after_state, error = self.execute(environment=planning_query.environment, action=child.action)
            if error:
                continue

            if self.visualize:
                self.visualize_example(action=child.action,
                                       after_state=after_state,
                                       before_state=before_state,
                                       before_state_predicted=tree.state,
                                       after_state_predicted=child.state,
                                       environment=planning_query.environment)

            # only include this example and continue the DFS if we were able to successfully execute the action
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
            yield from self.dfs(planner_params, planning_query, child, depth=depth + 1)

    def generate_example(self,
                         environment: Dict,
                         action: Dict,
                         before_state: Dict,
                         before_state_pred: Dict,
                         after_state: Dict,
                         after_state_pred: Dict,
                         classifier_start_t: int):
        if self.only_rejected_transitions and after_state_pred['num_diverged'].squeeze() != 1:
            return

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
        valid_out_examples_batched = add_model_error_and_filter(self.scenario,
                                                                actual=actual_batched,
                                                                predictions=predicted_batched,
                                                                out_example=example_batched,
                                                                labeling_params=self.labeling_params,
                                                                batch_size=1)
        test_shape = valid_out_examples_batched['time_idx'].shape[0]
        if test_shape == 1:
            valid_out_example = remove_batch(valid_out_examples_batched)

            yield valid_out_example  # yield here is more convenient than returning example/None
        elif test_shape > 1:
            raise NotImplementedError()
        else:
            return  # do nothing if there are no examples, i.e. test_shape == 0

    def execute(self, environment: Dict, action: Dict):
        self.service_provider.play()
        before_state = self.scenario.get_state()
        error = False
        try:
            self.scenario.execute_action(environment=environment, state=before_state, action=action)
        except RobotPlanningError:
            error = True
        after_state = self.scenario.get_state()
        return before_state, after_state, error

    def visualize_example(self,
                          action: Dict,
                          after_state: Dict,
                          before_state: Dict,
                          after_state_predicted: Dict,
                          before_state_predicted: Dict,
                          environment: Dict):
        before_state_predicted_viz = {add_predicted(k): v for k, v in before_state_predicted.items()}
        before_state_predicted_viz['joint_names'] = before_state_predicted['joint_names']
        after_state_predicted_viz = {add_predicted(k): v for k, v in after_state_predicted.items()}
        after_state_predicted_viz['joint_names'] = after_state_predicted['joint_names']

        self.scenario.plot_environment_rviz(environment)
        self.scenario.plot_state_rviz(before_state, idx=next(self.before_state_idx), label='before_actual')
        self.scenario.plot_state_rviz(before_state_predicted_viz, idx=next(self.before_state_pred_idx),
                                      label='before_predicted', color='blue')
        self.scenario.plot_state_rviz(after_state, idx=next(self.after_state_idx), label='after_actual')
        self.scenario.plot_state_rviz(after_state_predicted_viz, idx=next(self.after_state_pred_idx),
                                      label='after_predicted', color='blue')
        self.scenario.plot_action_rviz(before_state, action, idx=next(self.action_idx), label='')
        error = self.scenario.classifier_distance(after_state, after_state_predicted)
        self.scenario.plot_error_rviz(error)
        is_close = error < self.threshold
        self.scenario.plot_is_close(is_close)

    def clear_markers(self):
        self.scenario.reset_planning_viz()

    def on_gazebo_restarting(self, msg: Empty):
        self.restart = True


def store_bagfile():
    joint_state, links_states = get_states_to_save()
    bagfile_name = pathlib.Path(tempfile.NamedTemporaryFile().name)
    return save_test_scene_given_name(joint_state, links_states, bagfile_name, force=True)
