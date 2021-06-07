import pathlib
from typing import Dict, Optional

from colorama import Fore

from arc_utilities.algorithms import nested_dict_update
from link_bot_planning.analysis.results_utils import get_paths
from link_bot_planning.my_planner import PlanningResult, MyPlannerStatus
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.filepath_tools import load_hjson
from moonshine.moonshine_utils import numpify


def num_recovery_actions(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    count = 0
    for step in trial_datum['steps']:
        if step['type'] == 'executed_recovery':
            count += 1
    return count


def num_steps(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    paths = list(get_paths(trial_datum))
    return len(paths)


def cumulative_task_error(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    goal = trial_datum['goal']
    cumulative_error = 0
    for _, _, actual_state_t, _, _ in get_paths(trial_datum):
        cumulative_error += numpify(scenario.distance_to_goal(actual_state_t, goal))
    return cumulative_error


def task_error(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    goal = trial_datum['goal']
    final_actual_state = trial_datum['end_state']
    final_execution_to_goal_error = scenario.distance_to_goal(final_actual_state, goal)
    return numpify(final_execution_to_goal_error)


def success(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    final_execution_to_goal_error = task_error(scenario, trial_metadata, trial_datum)
    return final_execution_to_goal_error < trial_metadata['planner_params']['goal_params']['threshold']


def recovery_success(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    used_recovery = False
    _recovery_success = False
    for step in trial_datum['steps']:
        if step['type'] == 'executed_recovery':
            used_recovery = True
        if used_recovery and step['type'] != 'executed_recovery':
            _recovery_success = True
    return _recovery_success


def planning_time(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    planning_time = 0
    for step in trial_datum['steps']:
        planning_time += step['planning_result'].time
    return planning_time


def total_time(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    total_time = trial_datum['total_time']
    return total_time


def num_planning_attempts(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    attempts = 0
    for step in trial_datum['steps']:
        if step['type'] == 'executed_plan':
            attempts += 1
    return attempts


def any_solved(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    solved = False
    for step in trial_datum['steps']:
        if step['type'] == 'executed_plan':
            planning_result: PlanningResult = step['planning_result']
            if planning_result.status == MyPlannerStatus.Solved:
                solved = True
    return solved


def num_trials(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    return 1


def normalized_model_error(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    total_model_error = 0.0
    n_total_actions = 0
    for _, _, actual_state_t, planned_state_t, type_t in get_paths(trial_datum):
        if type_t == 'executed_plan' and planned_state_t is not None:
            model_error = scenario.classifier_distance(actual_state_t, planned_state_t)
            total_model_error += model_error
            n_total_actions += 1
    if n_total_actions == 0:
        return 0
    return total_model_error / n_total_actions


def load_analysis_params(analysis_params_filename: Optional[pathlib.Path] = None):
    analysis_params_common_filename = pathlib.Path("analysis_params/common.json")
    analysis_params = load_hjson(analysis_params_common_filename)

    if analysis_params_filename is not None:
        analysis_params = nested_dict_update(analysis_params, load_hjson(analysis_params_filename))

    return analysis_params


__all__ = [
    'num_trials',
    'task_error',
    'num_steps',
    'any_solved',
    'success',
    'total_time',
    'num_recovery_actions',
    'normalized_model_error',
    'num_planning_attempts',
    'cumulative_task_error',
    'recovery_success',
    'planning_time',

    'load_analysis_params',
]
