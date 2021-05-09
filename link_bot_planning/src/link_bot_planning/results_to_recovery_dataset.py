import pathlib
from time import perf_counter
from typing import Optional, List, Dict, Union

import numpy as np

import rospy
from link_bot_data.dataset_utils import tf_write_example, add_predicted
from link_bot_data.files_dataset import FilesDataset
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_planning.analysis import results_utils
from link_bot_planning.analysis.results_utils import NoTransitionsError, get_transitions
from link_bot_planning.results_to_classifier_dataset import compute_example_idx
from link_bot_pycommon.job_chunking import JobChunker
from link_bot_pycommon.marker_index_generator import marker_index_generator
from link_bot_pycommon.pycommon import try_make_dict_tf_float32
from moonshine.filepath_tools import load_hjson
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_tensors, add_batch_single, add_batch, remove_batch


class ResultsToRecoveryDataset:

    def __init__(self,
                 results_dir: pathlib.Path,
                 outdir: pathlib.Path,
                 labeling_params: Optional[Union[pathlib.Path, Dict]] = None,
                 trial_indices: Optional[List[int]] = None,
                 visualize: bool = False,
                 test_split: Optional[float] = None,
                 val_split: Optional[float] = None,
                 verbose: int = 1,
                 subsample_fraction: Optional[float] = None,
                 **kwargs):
        self.restart = False
        self.rng = np.random.RandomState(0)
        self.service_provider = GazeboServices()
        self.results_dir = results_dir
        self.outdir = outdir
        self.trial_indices = trial_indices
        self.subsample_fraction = subsample_fraction
        self.verbose = verbose

        if labeling_params is None:
            labeling_params = pathlib.Path('labeling_params/recovery/dual.json')

        if isinstance(labeling_params, Dict):
            self.labeling_params = labeling_params
        else:
            self.labeling_params = load_hjson(labeling_params)

        self.threshold = self.labeling_params['threshold']

        self.visualize = visualize
        self.scenario, self.metadata = results_utils.get_scenario_and_metadata(results_dir)

        self.example_idx = None
        self.files = FilesDataset(outdir, val_split, test_split)

        outdir.mkdir(exist_ok=True, parents=True)

    def run(self):
        results_utils.save_dynamics_dataset_hparams(self.results_dir, self.outdir, self.metadata)
        self.results_to_classifier_dataset()

    def results_to_classifier_dataset(self):
        logfilename = self.outdir / 'logfile.hjson'
        job_chunker = JobChunker(logfilename)

        t0 = perf_counter()
        last_t = t0
        total_examples = 0
        for trial_idx, datum in results_utils.trials_generator(self.results_dir, self.trial_indices):
            self.scenario.heartbeat()

            if job_chunker.has_result(str(trial_idx)):
                rospy.loginfo(f"Found existing recovery data for trial {trial_idx}")
                continue

            self.clear_markers()
            self.before_state_idx = marker_index_generator(0)
            self.before_state_pred_idx = marker_index_generator(1)
            self.after_state_idx = marker_index_generator(3)
            self.after_state_pred_idx = marker_index_generator(4)
            self.action_idx = marker_index_generator(5)

            example_idx_for_trial = 0

            self.example_idx = compute_example_idx(trial_idx, example_idx_for_trial)
            try:
                for example in self.result_datum_to_dynamics_dataset(datum, trial_idx, self.subsample_fraction):
                    now = perf_counter()
                    dt = now - last_t
                    total_dt = now - t0
                    last_t = now

                    self.example_idx = compute_example_idx(trial_idx, example_idx_for_trial)
                    total_examples += 1
                    if self.verbose >= 0:
                        msg = ' '.join([f'Trial {trial_idx}',
                                        f'Example {self.example_idx}',
                                        f'dt={dt:.3f},',
                                        f'total time={total_dt:.3f},',
                                        f'{total_examples=}'])
                        print(msg)
                    example = try_make_dict_tf_float32(example)
                    full_filename = tf_write_example(self.outdir, example, self.example_idx)
                    self.files.add(full_filename)
                    example_idx_for_trial += 1

                    job_chunker.store_result(trial_idx, {'trial':              trial_idx,
                                                         'examples for trial': example_idx_for_trial})
            except NoTransitionsError:
                rospy.logerr(f"Trial {trial_idx} had no transitions")
                pass

            job_chunker.store_result(trial_idx, {'trial':              trial_idx,
                                                 'examples for trial': example_idx_for_trial})

        self.files.split()

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

    def generate_example(self,
                         environment: Dict,
                         action: Dict,
                         before_state: Dict,
                         before_state_pred: Dict,
                         after_state: Dict,
                         after_state_pred: Dict,
                         classifier_start_t: int):
        # TODO: update that for outputting recovery examples

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
        test_shape = valid_out_examples_batched['time_idx'].shape[0]
        if test_shape == 1:
            valid_out_example = remove_batch(valid_out_examples_batched)

            yield valid_out_example
        elif test_shape > 1:
            raise NotImplementedError()
        else:
            pass  # do nothing if there are no examples, i.e. test_shape == 0

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

    def clear_markers(self):
        self.scenario.reset_planning_viz()
