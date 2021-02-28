import pathlib
from typing import Dict

import numpy as np

from link_bot_planning.my_planner import PlanningResult, MyPlannerStatus
from link_bot_planning.results_utils import get_paths
from link_bot_pycommon.experiment_scenario import ExperimentScenario


class ResultsMetric:
    def __init__(self, analysis_params, results_dir: pathlib.Path):
        super().__init__()
        self.analysis_params = analysis_params
        self.results_dir = results_dir
        self.values = {}
        self.method_indices = {}

    def setup_method(self, method_name: str, metadata: Dict):
        self.values[method_name] = []

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        raise NotImplementedError()

    def aggregate_trial(self, method_name: str, scenario: ExperimentScenario, trial_datum: Dict):
        metric_value = self.get_metric(scenario, trial_datum)
        self.values[method_name].append(metric_value)

    def aggregate_metric_values(self, method_name: str, metric_values):
        self.values[method_name].append(metric_values)

    def convert_to_numpy_arrays(self):
        for method_name, metric_values in self.values.items():
            self.values[method_name] = np.array(metric_values)


class TaskError(ResultsMetric):
    def __init__(self, analysis_params, results_dir: pathlib.Path):
        super().__init__(analysis_params, results_dir)
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


class NRecoveryActions(ResultsMetric):
    def __init__(self, analysis_params, results_dir: pathlib.Path):
        super().__init__(analysis_params, results_dir)

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        steps = trial_datum['steps']
        n_recovery = 0
        for step in steps:
            if step['type'] == 'executed_recovery':
                n_recovery += 1
        return n_recovery


class PercentageMERViolations(ResultsMetric):
    def __init__(self, analysis_params, results_dir: pathlib.Path):
        super().__init__(analysis_params, results_dir)

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        n_mer_violated = 0
        n_total_actions = 0
        _, actual_states, predicted_states, types = get_paths(trial_datum, scenario, False, 0)
        for actual_state_t, planned_state_t, type_t in zip(actual_states, predicted_states, types):
            if type_t == 'executed_plan':
                model_error = scenario.classifier_distance(actual_state_t, planned_state_t)
                mer_violated = model_error > self.analysis_params['mer_threshold']
                if mer_violated:
                    n_mer_violated += 1
                n_total_actions += 1
        return n_mer_violated / n_total_actions


class NMERViolations(ResultsMetric):
    def __init__(self, analysis_params, results_dir: pathlib.Path):
        super().__init__(analysis_params, results_dir)

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        n_mer_violated = 0
        _, actual_states, predicted_states, types = get_paths(trial_datum, scenario, False, 0)
        for actual_state_t, planned_state_t, type_t in zip(actual_states, predicted_states, types):
            if type_t == 'executed_plan':
                model_error = scenario.classifier_distance(actual_state_t, planned_state_t)
                mer_violated = model_error > self.analysis_params['mer_threshold']
                if mer_violated:
                    n_mer_violated += 1
        return n_mer_violated


class NPlanningAttempts(ResultsMetric):
    def __init__(self, analysis_params, results_dir: pathlib.Path):
        super().__init__(analysis_params, results_dir)

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        return len(trial_datum['steps'])


class TotalTime(ResultsMetric):
    def __init__(self, analysis_params, results_dir: pathlib.Path):
        super().__init__(analysis_params, results_dir)

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        return trial_datum['total_time']


class PlanningTime(ResultsMetric):
    def __init__(self, analysis_params, results_dir: pathlib.Path):
        super().__init__(analysis_params, results_dir)

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        steps = trial_datum['steps']
        planning_time = 0
        for step in steps:
            if step['type'] == 'executed_plan':
                planning_time += step['planning_result'].time
        return planning_time


class PlannerSolved(ResultsMetric):
    def __init__(self, analysis_params, results_dir: pathlib.Path):
        super().__init__(analysis_params, results_dir)

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        solved = False
        for step in trial_datum['steps']:
            if step['type'] == 'executed_plan':
                planning_result: PlanningResult = step['planning_result']
                if planning_result.status == MyPlannerStatus.Solved:
                    solved = True
        return solved


__all__ = [
    'ResultsMetric',
    'TaskError',
    'NRecoveryActions',
    'PercentageMERViolations',
    'NMERViolations',
    'NPlanningAttempts',
    'TotalTime',
    'PlanningTime',
    'PlannerSolved',
]
