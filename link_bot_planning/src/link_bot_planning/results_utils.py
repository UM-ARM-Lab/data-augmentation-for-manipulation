import json
import pathlib
import re
import threading
from time import sleep
from typing import Dict, Optional, List

import hjson

from link_bot_planning.my_planner import PlanningResult
from link_bot_planning.plan_and_execute import ExecutionResult
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.pycommon import paths_from_json
from link_bot_pycommon.serialization import load_gzipped_pickle, my_hdump
from moonshine.moonshine_utils import numpify


def fwd_model_params_from_planner_params(planner_params: Dict):
    fwd_model_dirs = paths_from_json(planner_params['fwd_model_dir'])
    representative_fwd_model_dir = fwd_model_dirs[0]
    fwd_hparams_filename = representative_fwd_model_dir.parent / 'params.json'
    with fwd_hparams_filename.open('r') as fwd_hparams_file:
        fwd_hparams = hjson.load(fwd_hparams_file)
    return fwd_hparams


def dynamics_dataset_params_from_classifier_params(classifier_params: Dict):
    dataset_dirs = paths_from_json(classifier_params['datasets'])
    representative_dataset_dir = dataset_dirs[0]
    dataset_hparams_filename = representative_dataset_dir / 'hparams.hjson'
    with dataset_hparams_filename.open('r') as dataset_hparams_file:
        dataset_hparams = hjson.load(dataset_hparams_file)
    return dataset_hparams


def dynamics_dataset_params_from_planner_params(planner_params: Dict):
    fwd_model_params = fwd_model_params_from_planner_params(planner_params)

    dataset_dirs = paths_from_json(fwd_model_params['datasets'])
    representative_dataset_dir = dataset_dirs[0]
    dataset_hparams_filename = representative_dataset_dir / 'hparams.hjson'
    with dataset_hparams_filename.open('r') as dataset_hparams_file:
        dataset_hparams = hjson.load(dataset_hparams_file)
    return dataset_hparams


def classifier_params_from_planner_params(planner_params):
    classifier_model_dirs = paths_from_json(planner_params['classifier_model_dir'])
    representative_classifier_model_dir = classifier_model_dirs[0]
    classifier_hparams_filename = representative_classifier_model_dir.parent / 'params.json'
    with classifier_hparams_filename.open('r') as classifier_hparams_file:
        classifier_hparams = hjson.load(classifier_hparams_file)
    return classifier_hparams


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


def get_paths(datum: Dict, scenario: ExperimentScenario, show_tree: bool = False, verbose: int = 0):
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


def get_scenario_and_metadata(results_dir: pathlib.Path):
    with (results_dir / 'metadata.json').open('r') as metadata_file:
        metadata_str = metadata_file.read()
        metadata = json.loads(metadata_str)
    scenario = get_scenario(metadata['scenario'])
    return scenario, metadata


def trials_generator(results_dir: pathlib.Path, trial_indices: Optional[List[int]]):
    if trial_indices is None:
        # assume we want all trials
        globbed_filenames = results_dir.glob("*.pkl.gz")
        filenames = []
        for filename in globbed_filenames:
            m = re.fullmatch(r'.*?([0-9]+)_metrics.pkl.gz', filename.as_posix())
            trial_idx = int(m.group(1))
            filenames.append((trial_idx, filename))
    else:
        filenames = []
        for trial_idx in trial_indices:
            filenames.append((trial_idx, results_dir / f'{trial_idx}_metrics.pkl.gz'))

    sorted_filenames = sorted(filenames)
    for trial_idx, results_filename in sorted_filenames:
        datum = load_gzipped_pickle(results_filename)
        yield trial_idx, datum


def save_dynamics_dataset_hparams(scenario: ExperimentScenario, results_dir: pathlib.Path, outdir: pathlib.Path,
                                  metadata: Dict):
    planner_params = metadata['planner_params']
    classifier_params = classifier_params_from_planner_params(planner_params)
    phase2_dataset_params = dynamics_dataset_params_from_classifier_params(classifier_params)
    dataset_hparams = phase2_dataset_params
    dataset_hparams_update = {
        'from_results':           results_dir,
        'seed':                   None,
        'data_collection_params': {
            'steps_per_traj': 2,
        },
    }
    dataset_hparams.update(dataset_hparams_update)
    with (outdir / 'hparams.hjson').open('w') as dataset_hparams_file:
        my_hdump(dataset_hparams, dataset_hparams_file, indent=2)


def edges_and_vertices_to_tree(edges: List[Dict], vertices: List[Dict]):
    pass
