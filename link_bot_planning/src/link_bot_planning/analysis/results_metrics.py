import pathlib
from typing import Dict, Optional, List, Iterable

import numpy as np
from colorama import Fore

from arc_utilities.algorithms import nested_dict_update
from link_bot_planning.analysis.results_utils import get_paths
from link_bot_planning.my_planner import PlanningResult, MyPlannerStatus
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.filepath_tools import load_hjson, load_json_or_hjson


class TrialMetrics:
    def __init__(self, analysis_params: Dict):
        super().__init__()
        self.analysis_params = analysis_params
        self.values = {}
        self.method_indices = {}
        self.metadatas = {}

    def setup_method(self, method_name: str, metadata: Dict):
        self.metadatas[method_name] = metadata
        self.values[method_name] = []

    def add_iteration(self, method_name: str):
        self.values[method_name].append([])

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        raise NotImplementedError()

    def aggregate_trial(self, method_name: str, scenario: ExperimentScenario, trial_datum: Dict):
        metric_value = self.get_metric(scenario, trial_datum)
        self.values[method_name].append(metric_value)

    def aggregate_trials(self, method_name: str):
        return np.mean(self.values[method_name])

    def after_all_trials(self, method_name: str):
        self.convert_to_numpy_arrays()

    def convert_to_numpy_arrays(self):
        for method_name, metric_values in self.values.items():
            self.values[method_name] = np.array(metric_values)


class TaskError(TrialMetrics):
    def __init__(self, analysis_params: Dict):
        super().__init__(analysis_params)
        self.goal_threshold = None

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        goal = trial_datum['goal']
        final_actual_state = trial_datum['end_state']
        final_execution_to_goal_error = scenario.distance_to_goal(final_actual_state, goal)
        return final_execution_to_goal_error

    def setup_method(self, method_name: str, metadata: Dict):
        super().setup_method(method_name, metadata)
        planner_params = metadata['planner_params']
        if 'goal_params' in planner_params:
            self.goal_threshold = planner_params['goal_params']['threshold']
        else:
            self.goal_threshold = planner_params['goal_threshold']


class Successes(TaskError):
    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        final_execution_to_goal_error = super().get_metric(scenario, trial_datum)
        return final_execution_to_goal_error < self.goal_threshold


class PercentSuccess(Successes):
    def __init__(self, analysis_params: Dict):
        super().__init__(analysis_params)

    def aggregate_trial(self, method_name: str, scenario: ExperimentScenario, trial_datum: Dict, iteration: int):
        metric_value = self.get_metric(scenario, trial_datum)
        self.values[method_name][iteration].append(metric_value)

    def aggregate_trials(self, method_name: str, iteration: int):
        mean = np.mean(self.values[method_name][iteration]) * 100
        self.values[method_name][iteration] = mean


class NRecoveryActions(TrialMetrics):
    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        steps = trial_datum['steps']
        n_recovery = 0
        for step in steps:
            if step['type'] == 'executed_recovery':
                n_recovery += 1
        return n_recovery


class PercentageMERViolations(TrialMetrics):
    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        n_mer_violated = 0
        n_total_actions = 0
        for _, actual_state_t, planned_state_t, type_t in get_paths(trial_datum):
            if type_t == 'executed_plan' and planned_state_t is not None:
                model_error = scenario.classifier_distance(actual_state_t, planned_state_t)
                mer_violated = model_error > self.analysis_params['mer_threshold']
                if mer_violated:
                    n_mer_violated += 1
                n_total_actions += 1
        if n_total_actions == 0:
            print(Fore.YELLOW + "no actions!?!")
            return 0
        return n_mer_violated / n_total_actions


class NMERViolations(TrialMetrics):
    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        n_mer_violated = 0
        for _, actual_state_t, planned_state_t, type_t in get_paths(trial_datum):
            if type_t == 'executed_plan' and planned_state_t is not None:
                model_error = scenario.classifier_distance(actual_state_t, planned_state_t)
                mer_violated = model_error > self.analysis_params['mer_threshold']
                if mer_violated:
                    n_mer_violated += 1
        return n_mer_violated


class NormalizedModelError(TrialMetrics):
    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        # NOTE: we could also normalize by action "size"?
        total_model_error = 0.0
        n_total_actions = 0
        for _, actual_state_t, planned_state_t, type_t in get_paths(trial_datum):
            if type_t == 'executed_plan' and planned_state_t is not None:
                model_error = scenario.classifier_distance(actual_state_t, planned_state_t)
                total_model_error += model_error
                n_total_actions += 1
        if n_total_actions == 0:
            print(Fore.YELLOW + "no actions!?!")
            return 0
        return total_model_error / n_total_actions


class NPlanningAttempts(TrialMetrics):
    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        return len(trial_datum['steps'])


class TotalTime(TrialMetrics):
    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        return trial_datum['total_time']


class PlanningTime(TrialMetrics):
    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        steps = trial_datum['steps']
        planning_time = 0
        for step in steps:
            if step['type'] == 'executed_plan':
                planning_time += step['planning_result'].time
        return planning_time


class PlannerSolved(TrialMetrics):
    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        return any_solved(trial_datum)


def any_solved(trial_datum: Dict):
    solved = False
    for step in trial_datum['steps']:
        if step['type'] == 'executed_plan':
            planning_result: PlanningResult = step['planning_result']
            if planning_result.status == MyPlannerStatus.Solved:
                solved = True
    return solved


def load_analysis_params(analysis_params_filename: Optional[pathlib.Path] = None):
    analysis_params_common_filename = pathlib.Path("analysis_params/common.json")
    analysis_params = load_hjson(analysis_params_common_filename)

    if analysis_params_filename is not None:
        analysis_params = nested_dict_update(analysis_params, load_hjson(analysis_params_filename))

    return analysis_params


def generate_multi_trial_metrics(analysis_params: Dict, results_dirs_dict: Dict):
    metrics = {}

    def _include_metric(metric: type):
        metrics[metric] = metric(analysis_params=analysis_params)

    # _include_metric(TaskError)
    # _include_metric(Successes)
    # _include_metric(NRecoveryActions)
    # _include_metric(TotalTime)
    # _include_metric(NPlanningAttempts)
    # _include_metric(NMERViolations)
    # _include_metric(NormalizedModelError)
    # _include_metric(PlanningTime)
    # _include_metric(PercentageMERViolations)
    # _include_metric(PlannerSolved)
    _include_metric(PercentSuccess)

    for method_name, (dirs, _) in results_dirs_dict.items():
        print(Fore.GREEN + f"processing {method_name} {[d.name for d in dirs]}")

        metadata = load_json_or_hjson(dirs[0].parent.parent, 'logfile')
        scenario = get_scenario(metadata['planner_params']['scenario'])
        for metric in metrics.values():
            metric.setup_method(method_name, metadata)

        for iteration, iteration_folder in enumerate(dirs):
            assert str(iteration) in iteration_folder.name  # sanity check

            for metric in metrics.values():
                metric.add_iteration(method_name)

            # NOTE: even though this is slow, parallelizing is not easy because "scenario" cannot be pickled
            metrics_filenames = list(iteration_folder.glob("*_metrics.pkl.gz"))
            for metrics_filename in metrics_filenames:
                datum = load_gzipped_pickle(metrics_filename)
                for metric in metrics.values():
                    metric.aggregate_trial(method_name, scenario, datum, iteration)

            for metric in metrics.values():
                metric.aggregate_trials(method_name, iteration)

    return metrics


def generate_per_trial_metrics(analysis_params: Dict, subfolders_ordered: List, method_names: Iterable):
    metrics = {}

    def _include_metric(metric: type):
        metrics[metric] = metric(analysis_params=analysis_params)

    _include_metric(TaskError)
    _include_metric(Successes)
    _include_metric(NRecoveryActions)
    _include_metric(TotalTime)
    _include_metric(NPlanningAttempts)
    _include_metric(NMERViolations)
    _include_metric(NormalizedModelError)
    _include_metric(PlanningTime)
    _include_metric(PercentageMERViolations)
    _include_metric(PlannerSolved)

    for subfolder, method_name in zip(subfolders_ordered, method_names):

        skip_filename = subfolder / '.skip'
        if skip_filename.exists():
            print(f"skipping {subfolder.name}")
        else:
            print(Fore.GREEN + f"processing {subfolder.name}")

        metrics_filenames = list(subfolder.glob("*_metrics.pkl.gz"))

        metadata = load_json_or_hjson(subfolder, 'metadata')

        scenario = get_scenario(metadata['scenario'])

        for metric in metrics.values():
            metric.setup_method(method_name, metadata)

        # NOTE: even though this is slow, parallelizing is not easy because "scenario" cannot be pickled
        for metrics_filename in metrics_filenames:
            datum = load_gzipped_pickle(metrics_filename)
            for metric in metrics.values():
                metric.aggregate_trial(method_name, scenario, datum)

        for metric in metrics.values():
            metric.after_all_trials(method_name)
    return metrics


__all__ = [
    'TrialMetrics',
    'TaskError',
    'Successes',
    'NRecoveryActions',
    'PercentageMERViolations',
    'NMERViolations',
    'NormalizedModelError',
    'NPlanningAttempts',
    'TotalTime',
    'PlanningTime',
    'PlannerSolved',
    'PercentSuccess',
    'load_analysis_params',
    'generate_per_trial_metrics',
]
