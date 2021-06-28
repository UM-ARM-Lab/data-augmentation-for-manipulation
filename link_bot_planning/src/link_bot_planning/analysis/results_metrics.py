import logging
import pathlib
from typing import Dict, Optional

import numpy as np

from arc_utilities.algorithms import nested_dict_update
from link_bot_planning.analysis.results_utils import get_paths, classifier_params_from_planner_params
from link_bot_planning.my_planner import PlanningResult, MyPlannerStatus
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.pycommon import has_keys
from moonshine.filepath_tools import load_hjson
from moonshine.moonshine_utils import numpify

logger = logging.getLogger(__file__)


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


def cumulative_planning_error(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    goal = trial_datum['goal']
    cumulative_error = 0
    for _, _, actual_state_t, _, _ in get_paths(trial_datum, full_path=True):
        cumulative_error += numpify(scenario.distance_to_goal(actual_state_t, goal))
    return cumulative_error


def task_error(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    goal = trial_datum['goal']
    final_actual_state = trial_datum['end_state']
    final_execution_to_goal_error = scenario.distance_to_goal(final_actual_state, goal)
    return numpify(final_execution_to_goal_error)


def success(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    final_execution_to_goal_error = task_error(scenario, trial_metadata, trial_datum)
    return int(final_execution_to_goal_error < trial_metadata['planner_params']['goal_params']['threshold'])


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


def mean_progagation_time(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    progagation_times = []
    # average across all the planning results in the trial
    for step in trial_datum['steps']:
        if 'planning_result' in step:
            dt = step['planning_result'].mean_propagate_time
            if dt is None:
                dt = np.nan
            progagation_times.append(dt)
    return np.mean(progagation_times)


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


def learned_classifier(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    c = trial_metadata['planner_params']['classifier_model_dir']
    found = False
    learned_classifier_ = None
    for c_i in c:
        if 'best_checkpoint' in c_i:
            if found:
                logger.warning("Multiple learned classifiers detected!!!")
            found = True
            learned_classifier_ = c_i
    return learned_classifier_


def classifier_source_env(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    cl_params = classifier_params_from_planner_params(trial_metadata['planner_params'])
    scene_name = has_keys(cl_params, ['classifier_dataset_hparams', 'scene_name'], None)
    if scene_name is None:
        print(f"Missing scene_name for {trial_metadata['planner_params']['classifier_model_dir'][0]}")
        return "no-scene-name"
    else:
        return scene_name


def target_env(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    return pathlib.Path(trial_metadata['test_scenes_dir']).name


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
    'cumulative_planning_error',
    'recovery_success',
    'planning_time',
    'learned_classifier',
    'classifier_source_env',
    'target_env',
    'mean_progagation_time',

    'load_analysis_params',
]
