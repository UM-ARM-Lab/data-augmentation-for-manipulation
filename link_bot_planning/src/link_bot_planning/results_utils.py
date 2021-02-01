import json
import threading
from time import sleep
from typing import Dict

from link_bot_planning.my_planner import PlanningResult
from link_bot_planning.plan_and_execute import ExecutionResult
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.pycommon import paths_from_json
from moonshine.moonshine_utils import numpify


def labeling_params_from_planner_params(planner_params, fallback_labeling_params: Dict):
    classifier_model_dirs = paths_from_json(planner_params['classifier_model_dir'])
    representative_classifier_model_dir = classifier_model_dirs[0]
    classifier_hparams_filename = representative_classifier_model_dir.parent / 'params.json'
    classifier_hparams = json.load(classifier_hparams_filename.open('r'))
    if 'labeling_params' in classifier_hparams:
        labeling_params = classifier_hparams['labeling_params']
    elif 'classifier_dataset_hparams' in classifier_hparams:
        labeling_params = classifier_hparams['classifier_dataset_hparams']['labeling_params']
    else:
        labeling_params = fallback_labeling_params
    return labeling_params


def get_paths(datum: Dict, scenario: ExperimentScenario, show_tree: bool=False, verbose: int=0):
    all_actual_states = []
    types = []
    all_predicted_states = []
    all_actions = []
    actual_states = None
    predicted_states = None

    steps = datum['steps']
    if len(steps) == 0:
        raise ValueError("no steps!")
    for step_idx, step in enumerate(steps):
        if verbose >= 1:
            print(step['type'])
        if step['type'] == 'executed_plan':
            planning_result: PlanningResult = step['planning_result']
            execution_result: ExecutionResult = step['execution_result']
            actions = planning_result.actions
            actual_states = execution_result.path
            predicted_states = planning_result.path
        elif step['type'] == 'executed_recovery':
            execution_result: ExecutionResult = step['execution_result']
            actions = [step['recovery_action']]
            actual_states = execution_result.path
            predicted_states = [None, None]
        else:
            raise NotImplementedError(f"invalid step type {step['type']}")

        actions = numpify(actions)
        actual_states = numpify(actual_states)
        predicted_states = numpify(predicted_states)

        all_actions.extend(actions)
        types.extend([step['type']] * len(actions))
        all_actual_states.extend(actual_states[:-1])
        all_predicted_states.extend(predicted_states[:-1])

        if show_tree and step['type'] == 'executed_plan':
            def _draw_tree_function(scenario, tree_json):
                print(f"n vertices {len(tree_json['vertices'])}")
                for vertex in tree_json['vertices']:
                    scenario.plot_tree_state(vertex, color='#77777722')
                    sleep(0.001)

            planning_result: PlanningResult = step['planning_result']
            tree_thread = threading.Thread(target=_draw_tree_function,
                                           args=(scenario, planning_result.tree,))
            tree_thread.start()
    # but do add the actual final states
    all_actual_states.append(actual_states[-1])
    all_predicted_states.append(predicted_states[-1])
    all_actions.append(all_actions[-1])
    types.append(types[-1])
    # also add the end_state, because due to rope settling it could be different
    all_actual_states.append(datum['end_state'])
    all_predicted_states.append(predicted_states[-1])  # just copy this again

    return all_actions, all_actual_states, all_predicted_states, types
