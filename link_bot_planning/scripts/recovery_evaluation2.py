#!/usr/bin/env python
import argparse
import pathlib

import numpy as np

from arc_utilities import ros_init
from arc_utilities.algorithms import nested_dict_update
from link_bot_classifiers.nn_recovery_policy import NNRecoveryPolicy2
from link_bot_data.dataset_utils import data_directory
from link_bot_gazebo import gazebo_services
from link_bot_pycommon.args import int_set_arg
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_trial


def evaluate_recovery(recovery_model_dir: pathlib.Path, nickname: pathlib.Path, trials, test_scenes: pathlib.Path,
                      seed: int, no_execution: bool,
                      on_exception: str, verbose: int):
    outdir = data_directory(pathlib.Path('results') / f"{nickname}-recovery-evaluation")

    _, params = load_trial(recovery_model_dir.parent.absolute())
    params_update = {
        'recovery':       {
            'recovery_model_dir': recovery_model_dir,
            'use_recovery':       True,
        },
        'scenario':       'dual_arm_rope_sim_val_with_robot_feasibility_checking',
        'max_step_size':  0.1,
        'real_time_rate': 0,
    }

    scenario = get_scenario(params['scenario'])
    rng = np.random.RandomState(0)
    recovery_policy = NNRecoveryPolicy2(recovery_model_dir, scenario, rng, params_update)

    service_provider = gazebo_services.GazeboServices()
    service_provider.play()

    service_provider.setup_env(verbose=verbose,
                               real_time_rate=params['real_time_rate'],
                               max_step_size=params['max_step_size'],
                               play=True)

    scenario.on_before_get_state_or_execute_action()

    for i in range(20):
        environment = scenario.get_environment(params)
        state = scenario.get_state()
        recovery_policy(environment=environment, state=state)

    scenario.robot.disconnect()


@ros_init.with_ros("recovery_evaluation")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recovery_model_dir", type=pathlib.Path)
    parser.add_argument("trials", type=int_set_arg, default="0-20")
    parser.add_argument("nickname", type=str, help='used in making the output directory')
    parser.add_argument("test_scenes", type=pathlib.Path)
    parser.add_argument("--seed", type=int, help='an additional seed for testing randomness', default=0)
    parser.add_argument("--no-execution", action="store_true", help='no execution')
    parser.add_argument("--on-exception", choices=['raise', 'catch', 'retry'], default='retry')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")

    args = parser.parse_args()

    # Start Services
    evaluate_recovery(**vars(args))


if __name__ == '__main__':
    main()
