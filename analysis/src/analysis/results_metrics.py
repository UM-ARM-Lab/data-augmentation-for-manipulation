import logging
import pathlib
from typing import Dict, Optional

import numpy as np
import rospkg

from analysis.results_utils import get_paths, classifier_params_from_planner_params, try_load_classifier_params
from arc_utilities.algorithms import nested_dict_update
from link_bot_planning.my_planner import PlanningResult, MyPlannerStatus
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.func_list_registrar import FuncListRegistrar
from link_bot_pycommon.pycommon import has_keys
from moonshine.filepath_tools import load_hjson
from moonshine.moonshine_utils import numpify

logger = logging.getLogger(__file__)

metrics_funcs = FuncListRegistrar()


@metrics_funcs
def num_recovery_actions(_: ExperimentScenario, __: Dict, trial_datum: Dict):
    count = 0
    for step in trial_datum['steps']:
        if step['type'] == 'executed_recovery':
            count += 1
    return count


@metrics_funcs
def num_steps(_: ExperimentScenario, __: Dict, trial_datum: Dict):
    paths = list(get_paths(trial_datum))
    return len(paths)


@metrics_funcs
def cumulative_task_error(scenario: ExperimentScenario, __: Dict, trial_datum: Dict):
    goal = trial_datum['goal']
    cumulative_error = 0
    for _, _, actual_state_t, _, _ in get_paths(trial_datum):
        cumulative_error += numpify(scenario.distance_to_goal(actual_state_t, goal))
    return cumulative_error


@metrics_funcs
def cumulative_planning_error(scenario: ExperimentScenario, __: Dict, trial_datum: Dict):
    goal = trial_datum['goal']
    cumulative_error = 0
    for _, _, actual_state_t, _, _ in get_paths(trial_datum, full_path=True):
        cumulative_error += numpify(scenario.distance_to_goal(actual_state_t, goal))
    return cumulative_error


@metrics_funcs
def task_error(scenario: ExperimentScenario, __: Dict, trial_datum: Dict):
    goal = trial_datum['goal']
    final_actual_state = trial_datum['end_state']
    final_execution_to_goal_error = scenario.distance_to_goal(final_actual_state, goal)
    return numpify(final_execution_to_goal_error)


@metrics_funcs
def is_fine_tuned(_: ExperimentScenario, trial_metadata: Dict, ___: Dict):
    try:
        classifier_model_dir = pathlib.Path(trial_metadata['planner_params']['classifier_model_dir'][0])
        classifier_hparams = try_load_classifier_params(classifier_model_dir)
        for k in list(classifier_hparams.keys()):
            if 'fine_tune' in k:
                return True
        return False
    except RuntimeError:
        return None

@metrics_funcs
def timeout(_: ExperimentScenario, trial_metadata: Dict, ___: Dict):
    return trial_metadata['planner_params']['termination_criteria']['timeout']


@metrics_funcs
def stop_on_error(_: ExperimentScenario, trial_metadata: Dict, ___: Dict):
    soe = has_keys(trial_metadata, ['planner_params', 'stop_on_error_above'], 999)
    return soe < 1


@metrics_funcs
def success(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    final_execution_to_goal_error = task_error(scenario, trial_metadata, trial_datum)
    return int(final_execution_to_goal_error < trial_metadata['planner_params']['goal_params']['threshold'])


@metrics_funcs
def recovery_success(_: ExperimentScenario, __: Dict, trial_datum: Dict):
    recovery_started = False
    recoveries_finished = 0
    recoveries_started = 0
    for i, step in enumerate(trial_datum['steps']):
        if recovery_started and step['type'] != 'executed_recovery':
            recoveries_finished += 1
            recovery_started = False
        elif step['type'] == 'executed_recovery' and not recovery_started:
            recoveries_started += 1
            recovery_started = True
    if recoveries_started == 0:
        _recovery_success = np.nan
    else:
        _recovery_success = np.divide(recoveries_finished, recoveries_started)
    return _recovery_success


@metrics_funcs
def planning_time(_: ExperimentScenario, __: Dict, trial_datum: Dict):
    _planning_time = 0
    for step in trial_datum['steps']:
        _planning_time += step['planning_result'].time
    return _planning_time


@metrics_funcs
def mean_progagation_time(_: ExperimentScenario, __: Dict, trial_datum: Dict):
    progagation_times = []
    # average across all the planning results in the trial
    for step in trial_datum['steps']:
        if 'planning_result' in step:
            dt = step['planning_result'].mean_propagate_time
            if dt is None:
                dt = np.nan
            progagation_times.append(dt)
    if len(progagation_times) == 0:
        return np.nan
    else:
        return np.mean(progagation_times)


@metrics_funcs
def total_time(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    total_time = trial_datum['total_time']
    return total_time


@metrics_funcs
def num_planning_attempts(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    attempts = 0
    for step in trial_datum['steps']:
        if step['type'] == 'executed_plan':
            attempts += 1
    return attempts


@metrics_funcs
def any_solved(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    solved = False
    for step in trial_datum['steps']:
        if step['type'] == 'executed_plan':
            planning_result: PlanningResult = step['planning_result']
            if planning_result.status == MyPlannerStatus.Solved:
                solved = True
    return solved


@metrics_funcs
def num_trials(_: ExperimentScenario, __: Dict, ___: Dict):
    return 1


@metrics_funcs
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


@metrics_funcs
def recovery_name(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    r = trial_metadata['planner_params']['recovery']
    use_recovery = r.get('use_recovery', False)
    if not use_recovery:
        return 'no-recovery'
    recovery_model_dir = r["recovery_model_dir"]
    return pathlib.Path(*pathlib.Path(recovery_model_dir).parent.parts[-2:]).as_posix()


@metrics_funcs
def classifier_name(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    c = trial_metadata['planner_params']['classifier_model_dir']
    found = False
    classifier_name_ = None
    for c_i in c:
        if 'best_checkpoint' in c_i:
            if found:
                logger.warning("Multiple learned classifiers detected!!!")
            found = True
            classifier_name_ = pathlib.Path(*pathlib.Path(c_i).parent.parts[-2:]).as_posix()
    if not found:
        if len(c) >= 1:
            classifier_name_ = c[0]
            found = True
        elif len(c) == 0:
            classifier_name_ = 'no classifier'
            found = True
    if not found:
        raise RuntimeError(f"Could not guess the classifier name:\n{c}")

    return classifier_name_


@metrics_funcs
def classifier_source_env(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    try:
        cl_params = classifier_params_from_planner_params(trial_metadata['planner_params'])
        scene_name = has_keys(cl_params, ['classifier_dataset_hparams', 'scene_name'], None)
        if scene_name is None:
            print(f"Missing scene_name for {trial_metadata['planner_params']['classifier_model_dir'][0]}")
            return "no-scene-name"
        else:
            return scene_name
    except RuntimeError:
        return None


@metrics_funcs
def target_env(scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    return pathlib.Path(trial_metadata['test_scenes_dir']).name


def load_analysis_params(analysis_params_filename: Optional[pathlib.Path] = None):
    analysis_params = load_analysis_hjson(pathlib.Path("analysis_params/common.json"))

    if analysis_params_filename is not None:
        analysis_params = nested_dict_update(analysis_params, load_hjson(analysis_params_filename))

    return analysis_params


def load_analysis_hjson(analysis_params_filename: pathlib.Path):
    r = rospkg.RosPack()
    analysis_dir = pathlib.Path(r.get_path("analysis"))
    analysis_params_common_filename = analysis_dir / analysis_params_filename
    analysis_params = load_hjson(analysis_params_common_filename)
    return analysis_params


metrics_names = [func.__name__ for func in metrics_funcs]
