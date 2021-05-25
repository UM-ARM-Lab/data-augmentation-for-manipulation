import pathlib
from typing import Dict, Optional

from colorama import Fore

from arc_utilities.algorithms import nested_dict_update
from link_bot_planning.analysis.results_utils import get_paths
from link_bot_planning.my_planner import PlanningResult, MyPlannerStatus
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.filepath_tools import load_hjson


def num_recovery_actions(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    count = 0
    for step in trial_datum['steps']:
        if step['type'] == 'executed_recovery':
            count += 1
    return count


def num_steps(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    paths = list(get_paths(trial_datum))
    return len(paths)


def task_error(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    goal = trial_datum['goal']
    final_actual_state = trial_datum['end_state']
    final_execution_to_goal_error = scenario.distance_to_goal(final_actual_state, goal)
    return final_execution_to_goal_error.numpy()


def success(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    final_execution_to_goal_error = task_error(scenario, trial_metadata, trial_datum)
    return final_execution_to_goal_error < trial_metadata['planner_params']['goal_params']['threshold']


def total_time(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    total_time = trial_datum['total_time']
    return total_time


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
        print(Fore.YELLOW + "no actions!?!")
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

    'load_analysis_params',
]
