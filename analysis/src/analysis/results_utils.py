import logging
import os
import pathlib
import re
from typing import Dict, Optional, List, Union

import hjson
import numpy as np
import wandb
from colorama import Fore
from matplotlib import cm

import rospy
from arc_utilities.algorithms import zip_repeat_shorter
from link_bot_classifiers.classifier_utils import is_torch_model, strip_torch_model_prefix
from link_bot_planning.my_planner import PlanningResult
from link_bot_planning.trial_result import ExecutionResult, planning_trial_name
from link_bot_pycommon.get_scenario import get_scenario_cached
from link_bot_pycommon.grid_utils_np import extent_res_to_origin_point
from link_bot_pycommon.pycommon import paths_from_json, has_keys
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from link_bot_pycommon.screen_recorder import ScreenRecorder
from link_bot_pycommon.serialization import load_gzipped_pickle, my_hdump
from link_bot_pycommon.wandb_utils import reformat_run_config_dict
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.filepath_tools import load_params, load_json_or_hjson
from moonshine.numpify import numpify

logger = logging.getLogger(__name__)


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
    classifier_hparams = try_load_classifier_params(representative_classifier_model_dir)
    return classifier_hparams


def try_load_classifier_params(representative_classifier_model_dir, parent=pathlib.Path('.')):
    if is_torch_model(representative_classifier_model_dir):  # this is a pytorch model
        run_id = strip_torch_model_prefix(representative_classifier_model_dir)
        api = wandb.Api()
        run = api.run(f"armlab/mde/{run_id}")
        return reformat_run_config_dict(run.config)

    p1 = representative_classifier_model_dir.parent
    p2 = pathlib.Path(*p1.parts[2:])
    p3 = pathlib.Path('/media/shared/ift') / p2
    paths_to_try = [
        p1,
        pathlib.Path('/media/shared/') / representative_classifier_model_dir.parent,
        p3,
        parent / p1,
    ]
    for path_to_try in paths_to_try:
        if path_to_try.exists():
            classifier_hparams = load_params(path_to_try)
            return classifier_hparams
    return None


def classifer_dataset_params_from_planner_params(planner_params: Dict):
    classifier_params = classifier_params_from_planner_params(planner_params)

    dataset_dirs = paths_from_json(classifier_params['datasets'])
    representative_dataset_dir = dataset_dirs[0]
    dataset_hparams = load_params(representative_dataset_dir)
    return dataset_hparams


def labeling_params_from_planner_params(planner_params, fallback_labeling_params: Dict):
    classifier_model_dirs = paths_from_json(planner_params['classifier_model_dir'])
    representative_classifier_model_dir = classifier_model_dirs[0]
    classifier_hparams = try_load_classifier_params(representative_classifier_model_dir)

    if 'threshold' in classifier_hparams:
        labeling_params = {'threshold': classifier_hparams['threshold']}
    elif 'labeling_params' in classifier_hparams:
        labeling_params = classifier_hparams['labeling_params']
    else:
        labeling_params = has_keys(classifier_hparams, ['classifier_dataset_hparams', 'labeling_params'])
        if not labeling_params:
            labeling_params = fallback_labeling_params
    return labeling_params


def get_paths(datum: Dict, verbose: int = 0, full_path: bool = True):
    steps = datum['steps']

    if len(steps) == 0:
        return

    types = []
    actions = None
    actual_states = None
    predicted_states = None
    e = None
    for step_idx, step in enumerate(steps):
        e = step['planning_query'].environment

        if verbose >= 1:
            logger.debug(step['type'])
        if step['type'] == 'executed_plan':
            planning_result: PlanningResult = step['planning_result']
            execution_result: ExecutionResult = step['execution_result']
            actions = planning_result.actions
            actual_states = execution_result.path
            predicted_states = planning_result.path
            if verbose >= 2:
                print(planning_result.status)
        elif step['type'] == 'executed_recovery':
            execution_result: ExecutionResult = step['execution_result']
            actions = [step['recovery_action']]
            actual_states = execution_result.path
            predicted_states = [None, None]
        else:
            raise NotImplementedError(f"invalid step type {step['type']}")

        if len(actions) == 0 or actions[0] is None:
            logger.info("Skipping step with no actions")
            continue
        actions = numpify(actions)
        actual_states = numpify(actual_states)
        predicted_states = numpify(predicted_states)

        types = [step['type']] * len(actions)
        j = range(len(actions) + 1)
        if full_path:
            full_path_for_step = zip_repeat_shorter(actions, actual_states, predicted_states, types, j)
            yield from [(e, *p_t) for p_t in full_path_for_step]
        else:
            actions_last_rep = actions + [actions[-1]]
            types_last_rep = types + [types[-1]]
            path_for_step = zip(actions_last_rep, actual_states, predicted_states, types_last_rep, j)
            yield from [(e, *p_t) for p_t in path_for_step]

    # but do add the actual final states
    # e will be whatever the environment from the last step was
    if len(actions) > 0 and actions[0] is not None:
        yield e, actions[-1], actual_states[-1], predicted_states[-1], types[-1], -1

    if len(types) > 0:
        yield e, actions[-1], datum['end_state'], predicted_states[-1], types[-1], -1


def get_recovery_transitions(datum: Dict):
    paths = get_paths(datum, full_path=False)
    next_paths = get_paths(datum, full_path=False)
    try:
        next(next_paths)

        for before, after in zip(paths, next_paths):
            e, action, before_state, _, before_type, _ = before
            _, _, after_state, _, _, _ = after
            if before_type == 'executed_recovery':
                yield e, action, before_state, after_state, before_type

    except StopIteration:
        raise NoTransitionsError()


def get_transitions(datum: Dict):
    steps = datum['steps']

    if len(steps) == 0:
        raise NoTransitionsError()

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
            logger.info("Skipping step with no actions")
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
    scenario_params = has_keys(metadata, ['planner_params', 'scenario_params'])
    if not scenario_params:
        scenario_params = {'rope_name': 'rope_3d'}
    scenario_params = dict(scenario_params)
    scenario = get_scenario_cached(metadata['scenario'], scenario_params)
    return scenario, metadata


def trials_generator(results_dir: pathlib.Path, trials: Optional[List[int]] = None):
    idx_and_filenames = trials_filenames_generator(results_dir, trials)

    for trial_idx, datum_filename in idx_and_filenames:
        datum = load_gzipped_pickle(datum_filename)
        yield trial_idx, datum, datum_filename


def trials_filenames_generator(results_dir, trials: Optional = None):
    if trials is None:
        # assume we want all trials
        idx_and_filenames = list_all_planning_results_trials(results_dir)
    else:
        idx_and_filenames = []
        for trial_idx in trials:
            idx_and_filenames.append((trial_idx, results_dir / planning_trial_name(trial_idx)))

    return sorted(idx_and_filenames)


def list_numbered_files(results_dir, pattern, extension):
    globbed_filenames = results_dir.glob(f"*.{extension}")
    filenames = []
    for filename in globbed_filenames:
        m = re.fullmatch(pattern + extension, filename.as_posix())
        trial_idx = int(m.group(1))
        filenames.append((trial_idx, filename))
    return sorted(filenames)


def list_all_planning_results_trials(results_dir):
    return list_numbered_files(results_dir, extension='pkl.gz', pattern=r'.*?([0-9]+)_metrics.')


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
               full_plan: bool,
               screen_recorder: Optional[ScreenRecorder] = None):
    if screen_recorder is not None:
        screen_recorder.start()

    planner_params = metadata['planner_params']
    goal_threshold = get_goal_threshold(planner_params)

    labeling_params = labeling_params_from_planner_params(planner_params, fallback_labeing_params)

    steps = datum['steps']

    if len(steps) == 0:
        rospy.logerr("zero steps!?")

    goal = datum['goal']
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

    scenario.reset_viz()
    while not anim.done:
        t = anim.t()
        e_t, a_t, s_t, s_t_pred, type_t, j_t = paths[t]

        actual_state_color = '#00000088' if j_t != 0 else '#ffffffaa'

        if 'scene_msg' in e_t and 'attached_collision_objects' not in s_t:
            s_t['attached_collision_objects'] = e_t['scene_msg'].robot_state.attached_collision_objects
        if 'origin_point' not in e_t:
            e_t['origin_point'] = extent_res_to_origin_point(e_t['extent'], e_t['res'])
        scenario.plot_environment_rviz(e_t)
        scenario.plot_state_rviz(s_t, label='actual', color=actual_state_color)

        c = 'r'
        if s_t_pred is not None:
            if 'error' in s_t_pred:
                pred_error = np.squeeze(s_t_pred['error'])
                scenario.plot_pred_error_rviz(pred_error)
                c = cm.jet_r(pred_error)
            else:
                scenario.plot_pred_error_rviz(-999)

            if 'accept_probability' in s_t_pred:
                accept_probability_t = np.squeeze(s_t_pred['accept_probability'])
                scenario.plot_accept_probability(accept_probability_t)
                c = cm.jet_r(accept_probability_t)
            else:
                scenario.plot_accept_probability(-999)

        if t < anim.max_t:
            action_color = _type_action_color(type_t)
            scenario.plot_action_rviz(s_t, a_t, color=action_color)

        if s_t_pred is not None:
            if 'scene_msg' in e_t and 'attached_collision_objects' not in s_t_pred:
                s_t_pred['attached_collision_objects'] = e_t['scene_msg'].robot_state.attached_collision_objects
            scenario.plot_state_rviz(s_t_pred, label='predicted', color=c)
            is_close = scenario.compute_label(s_t, s_t_pred, labeling_params)
            scenario.plot_is_close(is_close)
            model_error = scenario.classifier_distance(s_t, s_t_pred)
            scenario.plot_error_rviz(model_error)
        else:
            scenario.plot_is_close(None)
            scenario.plot_error_rviz(-1)

        dist_to_goal = scenario.distance_to_goal(s_t, goal)
        actually_at_goal = dist_to_goal < goal_threshold
        scenario.plot_goal_rviz(goal, goal_threshold, actually_at_goal)

        anim.step()

    if screen_recorder is not None:
        screen_recorder.stop()


def get_all_results_subdirs(dirs: Union[pathlib.Path, List[pathlib.Path]], regenerate: bool):
    if isinstance(dirs, pathlib.Path):
        dirs = [dirs]

    results_subdirs = []
    for dir in dirs:
        cache_filename = dir / 'results_subdirs.txt'
        if regenerate:
            results_subdirs_in_dir = get_results_subdirs_in_dir(dir)
            with cache_filename.open('w') as cache_f:
                for d in results_subdirs_in_dir:
                    cache_f.write(d.as_posix())
                    cache_f.write('\n')
        else:
            if cache_filename.exists():
                with cache_filename.open("r") as cache_f:
                    results_subdirs_in_dir = [pathlib.Path(l.strip("\n")) for l in cache_f.readlines()]
            else:
                results_subdirs_in_dir = get_results_subdirs_in_dir(dir)
                with cache_filename.open('w') as cache_f:
                    for d in results_subdirs_in_dir:
                        cache_f.write(d.as_posix())
                        cache_f.write('\n')

        results_subdirs.extend(results_subdirs_in_dir)

    return results_subdirs


def get_results_subdirs_in_dir(dir):
    results_subdirs_in_dir = []
    for (root, dirs, files) in os.walk(dir.as_posix()):
        if 'classifier_datasets' in root or root == 'training_logdir':
            continue
        for f in files:
            if '_metrics.pkl.gz' in f:
                results_subdirs_in_dir.append(pathlib.Path(root))
                break
    return results_subdirs_in_dir


def dataset_dir_to_num_examples(p):
    p = pathlib.Path(p)
    for part in p.parts:
        m = re.match(r'.*examples_(\d+)', part)
        if m:
            i = int(m.group(1))
            return i
    return -1


def dataset_dir_to_iter(p):
    p = pathlib.Path(p)
    for part in p.parts:
        m = re.match(r'iter.*?_(\d+)', part)
        if m:
            i = int(m.group(1))
            return i
    return -1


def classifier_name_to_iter(p):
    p = pathlib.Path(p)
    for part in p.parts:
        if 'untrained-1' in part:
            return 0
        m = re.match(r'iteration_(\d+)_classifier_training_logdir', part)
        if m:
            i = int(m.group(1))
            return i
    return -1
