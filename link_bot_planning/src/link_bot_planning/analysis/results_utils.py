import pathlib
import re
from typing import Dict, Optional, List

import hjson
from colorama import Fore

import rospy
from arc_utilities.algorithms import zip_repeat_shorter
from link_bot_planning.my_planner import PlanningResult, PlanningQuery
from link_bot_planning.plan_and_execute import ExecutionResult
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.pycommon import paths_from_json
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from link_bot_pycommon.serialization import load_gzipped_pickle, my_hdump
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.filepath_tools import load_params, load_json_or_hjson
from moonshine.moonshine_utils import numpify


class NoTransitionsError(Exception):
    pass

def fwd_model_params_from_planner_params(planner_params: Dict):
    fwd_model_dirs = paths_from_json(planner_params['fwd_model_dir'])
    representative_fwd_model_dir = fwd_model_dirs[0]
    fwd_hparams = load_params(representative_fwd_model_dir.parent)
    return fwd_hparams


def dynamics_dataset_params_from_classifier_params(classifier_params: Dict):
    dataset_dirs = paths_from_json(classifier_params['datasets'])
    representative_dataset_dir = dataset_dirs[0]
    dataset_hparams = load_params(representative_dataset_dir)
    return dataset_hparams


def dynamics_dataset_params_from_planner_params(planner_params: Dict):
    fwd_model_params = fwd_model_params_from_planner_params(planner_params)

    dataset_dirs = paths_from_json(fwd_model_params['datasets'])
    representative_dataset_dir = dataset_dirs[0]
    dataset_hparams = load_params(representative_dataset_dir)
    return dataset_hparams


def classifier_params_from_planner_params(planner_params):
    classifier_model_dirs = paths_from_json(planner_params['classifier_model_dir'])
    representative_classifier_model_dir = classifier_model_dirs[0]
    classifier_hparams = load_params(representative_classifier_model_dir.parent)
    return classifier_hparams


def classifer_dataset_params_from_planner_params(planner_params: Dict):
    classifier_params = classifier_params_from_planner_params(planner_params)

    dataset_dirs = paths_from_json(classifier_params['datasets'])
    representative_dataset_dir = dataset_dirs[0]
    dataset_hparams = load_params(representative_dataset_dir)
    return dataset_hparams


def labeling_params_from_planner_params(planner_params, fallback_labeling_params: Dict):
    classifier_model_dirs = paths_from_json(planner_params['classifier_model_dir'])
    representative_classifier_model_dir = classifier_model_dirs[0]
    classifier_hparams_filename = representative_classifier_model_dir.parent
    classifier_hparams = load_params(classifier_hparams_filename)
    if 'labeling_params' in classifier_hparams:
        labeling_params = classifier_hparams['labeling_params']
    elif 'classifier_dataset_hparams' in classifier_hparams:
        labeling_params = classifier_hparams['classifier_dataset_hparams']['labeling_params']
    else:
        labeling_params = fallback_labeling_params
    return labeling_params


def get_paths(datum: Dict, verbose: int = 0, full_path: bool = True):
    steps = datum['steps']

    if len(steps) == 0:
        return

    types = []
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

        if len(actions) == 0 or actions[0] is None:
            print("Skipping step with no actions")
            continue
        actions = numpify(actions)
        actual_states = numpify(actual_states)
        predicted_states = numpify(predicted_states)

        types = [step['type']] * len(actions)
        if full_path:
            yield from zip_repeat_shorter(actions, actual_states, predicted_states, types)
        else:
            yield from zip(actions, actual_states, predicted_states, types)

    # but do add the actual final states
    if len(actions) > 0 and actions[0] is not None:
        yield actions[-1], actual_states[-1], predicted_states[-1], types[-1]

    if len(types) > 0:
        yield actions[-1], datum['end_state'], predicted_states[-1], types[-1]


def get_transitions(datum: Dict):
    steps = datum['steps']

    if len(steps) == 0:
        raise NoTransitionsError(steps)

    for step_idx, step in enumerate(steps):
        if step['type'] == 'executed_plan':
            planning_result: PlanningResult = step['planning_result']
            execution_result: ExecutionResult = step['execution_result']
            actions = planning_result.actions
            actual_states = execution_result.path
            predicted_states = planning_result.path
        elif step['type'] == 'executed_recovery':
            continue
        else:
            raise NotImplementedError(f"invalid step type {step['type']}")

        if len(actions) == 0 or actions[0] is None:
            print("Skipping step with no actions")
            continue
        actions = numpify(actions)
        actual_states = numpify(actual_states)
        predicted_states = numpify(predicted_states)

        e = step['planning_query'].environment
        types = [step['type']] * len(actions)
        n_actual_states = len(actual_states)

        for t in range(n_actual_states - 1):
            before_state_pred_t = predicted_states[t]
            before_state_t = actual_states[t]
            after_state_pred_t = predicted_states[t + 1]
            after_state_t = actual_states[t + 1]
            a_t = actions[t]
            type_t = types[t]
            yield e, (before_state_pred_t, before_state_t), a_t, (after_state_pred_t, after_state_t), type_t


def get_scenario_and_metadata(results_dir: pathlib.Path):
    metadata = load_json_or_hjson(results_dir, 'metadata')
    scenario = get_scenario(metadata['scenario'])
    return scenario, metadata


def trials_generator(results_dir: pathlib.Path, trial_indices: Optional[List[int]] = None):
    if trial_indices is None:
        # assume we want all trials
        filenames = list_all_planning_results_trials(results_dir)
    else:
        filenames = []
        for trial_idx in trial_indices:
            filenames.append((trial_idx, results_dir / f'{trial_idx}_metrics.pkl.gz'))

    sorted_filenames = sorted(filenames)
    for trial_idx, results_filename in sorted_filenames:
        datum = load_gzipped_pickle(results_filename)
        yield trial_idx, datum


def list_numbered_files(results_dir, pattern, extension):
    globbed_filenames = results_dir.glob(f"*.{extension}")
    filenames = []
    for filename in globbed_filenames:
        m = re.fullmatch(pattern + extension, filename.as_posix())
        trial_idx = int(m.group(1))
        filenames.append((trial_idx, filename))
    return filenames


def list_all_planning_results_trials(results_dir):
    return list_numbered_files(results_dir, extension='pkl.gz', pattern=r'.*?([0-9]+)_metrics.')


def save_dynamics_dataset_hparams(results_dir: pathlib.Path, outdir: pathlib.Path, metadata: Dict):
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


def print_percentage(description: str, numerator: int, denominator: int):
    if denominator == 0:
        print(f'{description:80s} {numerator}/0 (division by zero)')
    else:
        print(f'{description:80s} {numerator}/{denominator}, {numerator / denominator * 100:.1f}%')


def save_order(outdir: pathlib.Path, subfolders_ordered: List[pathlib.Path]):
    sort_order_filename = outdir / 'sort_order.txt'
    with sort_order_filename.open("w") as sort_order_file:
        my_hdump(subfolders_ordered, sort_order_file)


def load_sort_order(outdir: pathlib.Path, unsorted_dirs: List[pathlib.Path]):
    sort_order_filename = outdir / 'sort_order.txt'
    if sort_order_filename.exists():
        with sort_order_filename.open("r") as sort_order_file:
            subfolders_ordered = hjson.load(sort_order_file)
        subfolders_ordered = paths_from_json(subfolders_ordered)
        return subfolders_ordered
    return unsorted_dirs


def load_order(prompt_order: bool, directories: List[pathlib.Path], out_dir: pathlib.Path):
    if prompt_order:
        for idx, results_dir in enumerate(directories):
            print("{}) {}".format(idx, results_dir))
        sort_order = input(Fore.CYAN + "Enter the desired order:\n" + Fore.RESET)
        dirs_ordered = [directories[int(i)] for i in sort_order.split(' ')]
        save_order(out_dir, dirs_ordered)
    else:
        dirs_ordered = load_sort_order(out_dir, directories)
    return dirs_ordered


def add_number_to_method_name(method_name: str):
    if method_name[-1].isnumeric():
        i = int(method_name[-1])
        return method_name[:-1] + str(i + 1)
    else:
        return method_name + "2"


def get_goal_threshold(planner_params):
    if 'goal_params' in planner_params:
        goal_threshold = planner_params['goal_params']['threshold']
    else:
        goal_threshold = planner_params['goal_threshold']
    return goal_threshold


def plot_steps(scenario: ScenarioWithVisualization,
               datum: Dict,
               metadata: Dict,
               fallback_labeing_params: Dict,
               verbose: int,
               full_plan: bool):
    planner_params = metadata['planner_params']
    goal_threshold = get_goal_threshold(planner_params)

    labeling_params = labeling_params_from_planner_params(planner_params, fallback_labeing_params)

    steps = datum['steps']

    if len(steps) == 0:
        q: PlanningQuery = datum['planning_queries'][0]
        start = q.start
        goal = q.goal
        environment = q.environment
        anim = RvizAnimationController(n_time_steps=1)
        scenario.plot_state_rviz(start, label='actual', color='#ff0000aa')
        scenario.plot_goal_rviz(goal, goal_threshold)
        scenario.plot_environment_rviz(environment)
        anim.step()
        return

    goal = datum['goal']
    first_step = steps[0]
    planning_query: PlanningQuery = first_step['planning_query']
    environment = planning_query.environment
    paths = list(get_paths(datum, verbose, full_plan))

    if len(paths) == 0:
        rospy.logwarn("empty trial!")
        return

    anim = RvizAnimationController(n_time_steps=len(paths))

    def _type_action_color(type_t: str):
        if type_t == 'executed_plan':
            return 'b'
        elif type_t == 'executed_recovery':
            return '#ff00ff'

    scenario.reset_planning_viz()
    while not anim.done:
        scenario.plot_environment_rviz(environment)
        t = anim.t()
        a_t, s_t, s_t_pred, type_t = paths[t]
        scenario.plot_state_rviz(s_t, label='actual', color='#ff0000aa')
        c = '#0000ffaa'
        if t < anim.max_t:
            action_color = _type_action_color(type_t)
            scenario.plot_action_rviz(s_t, a_t, color=action_color)

        if s_t_pred is not None:
            scenario.plot_state_rviz(s_t_pred, label='predicted', color=c)
            is_close = scenario.compute_label(s_t, s_t_pred, labeling_params)
            scenario.plot_is_close(is_close)
        else:
            scenario.plot_is_close(None)

        dist_to_goal = scenario.distance_to_goal(s_t, goal)
        actually_at_goal = dist_to_goal < goal_threshold
        scenario.plot_goal_rviz(goal, goal_threshold, actually_at_goal)

        anim.step()