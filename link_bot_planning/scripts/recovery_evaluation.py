#!/usr/bin/env python
import argparse
import logging
import pathlib
import warnings

import tensorflow as tf

from analysis import results_metrics
from link_bot_planning.my_planner import PlanningResult, MyPlannerStatus, PlanningQuery
from link_bot_planning.trial_result import ExecutionResult
from link_bot_planning.timeout_or_not_progressing import EvalRecoveryPTC
from link_bot_pycommon.serialization import load_gzipped_pickle

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    from ompl import util as ou

from arc_utilities import ros_init
from arc_utilities.algorithms import nested_dict_update
from link_bot_data.dataset_utils import make_unique_outdir
from link_bot_gazebo import gazebo_services
from link_bot_planning.get_planner import get_planner
from link_bot_planning.planning_evaluation import EvaluatePlanning, load_planner_params
from link_bot_pycommon.args import int_set_arg
from link_bot_pycommon.job_chunking import JobChunker
from moonshine.filepath_tools import load_params


def no_recovery(planning_result: PlanningResult, execution_result: ExecutionResult):
    return planning_result.status != MyPlannerStatus.NotProgressing


def evaluate_recovery(recovery_model_dir: pathlib.Path, planner_params_filename: pathlib.Path, nickname: pathlib.Path,
                      trials,
                      test_scenes: pathlib.Path,
                      seed: int, no_execution: bool,
                      on_exception: str, verbose: int):
    outdir = make_unique_outdir(pathlib.Path('results') / f"{nickname}-recovery-evaluation")

    planner_params = load_planner_params(planner_params_filename)
    recovery_model_hparams = load_params(recovery_model_dir)
    classifier_model_checkpoint = pathlib.Path(recovery_model_hparams['recovery_dataset_hparams']['classifier_model'])
    params_update = {
        'recovery':             {
            'recovery_model_dir': recovery_model_dir / 'best_checkpoint',
        },
        'classifier_model_dir': [
            classifier_model_checkpoint,
            'cl_trials/new_feasibility_baseline/none'
        ],
        'n_shortcut_attempts':  0,
        'termination_criteria': {
            'max_attempts': 20,
            'timeout':      20,
        }
    }
    planner_params = nested_dict_update(planner_params, params_update)

    service_provider = gazebo_services.GazeboServices()
    service_provider.play()
    planner = get_planner(planner_params=planner_params, verbose=verbose, log_full_tree=False)

    service_provider.setup_env(verbose=verbose,
                               real_time_rate=planner_params['real_time_rate'],
                               max_step_size=planner.fwd_model.max_step_size,
                               play=True)

    planner.scenario.on_before_get_state_or_execute_action()

    def _override_ptc(planning_query: PlanningQuery):
        return EvalRecoveryPTC(planning_query, planner_params['termination_criteria'], 0)

    planner.make_ptc = _override_ptc

    job_chunker = JobChunker(logfile_name=outdir / 'logfile.hjson')
    extra_end_conditions = [
        no_recovery
    ]
    runner = EvaluatePlanning(planner=planner,
                              service_provider=service_provider,
                              job_chunker=job_chunker,
                              trials=trials,
                              verbose=verbose,
                              planner_params=planner_params,
                              outdir=outdir,
                              no_execution=no_execution,
                              test_scenes_dir=test_scenes,
                              seed=seed,
                              extra_end_conditions=extra_end_conditions
                              )

    runner.run()
    planner.scenario.robot.disconnect()

    n_trials = len(trials)
    metrics_filenames = list(outdir.glob("*_metrics.pkl.gz"))
    n_success = 0
    num_recovery_actions = 0
    for file_idx, metrics_filename in enumerate(metrics_filenames):
        trial_datum = load_gzipped_pickle(metrics_filename)
        if results_metrics.recovery_success(planner.scenario, {}, trial_datum):
            n_success += 1
        num_recovery_actions += results_metrics.num_recovery_actions(planner.scenario, {}, trial_datum)

    print(classifier_model_checkpoint.parent)
    print(recovery_model_dir)
    print(num_recovery_actions, n_success, n_trials)


@ros_init.with_ros("recovery_evaluation")
def main():
    ou.setLogLevel(ou.LOG_ERROR)
    tf.get_logger().setLevel(logging.ERROR)
    tf.autograph.set_verbosity(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("recovery_model_dir", type=pathlib.Path)
    parser.add_argument("planner_params_filename", type=pathlib.Path)
    parser.add_argument("trials", type=int_set_arg, default="0-20")
    parser.add_argument("nickname", type=str, help='used in making the output directory')
    parser.add_argument("test_scenes", type=pathlib.Path)
    parser.add_argument("--seed", type=int, help='an additional seed for testing randomness', default=0)
    parser.add_argument("--no-execution", action="store_true", help='no execution')
    parser.add_argument("--on-exception", choices=['raise', 'catch', 'retry'], default='retry')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")

    args = parser.parse_args()

    evaluate_recovery(**vars(args))


if __name__ == '__main__':
    main()
