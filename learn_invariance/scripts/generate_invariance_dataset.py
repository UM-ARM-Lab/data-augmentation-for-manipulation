#!/usr/bin/env python
import argparse
import pathlib
from itertools import cycle

import hjson
import numpy as np
import transformations
from colorama import Fore
from progressbar import progressbar

from arc_utilities import ros_init
from learn_invariance.new_dynamics_dataset_loader import NewDynamicsDatasetLoader
from learn_invariance.transform_link_states import transform_link_states
from link_bot_data import base_dataset
from link_bot_data.dataset_utils import pkl_write_example, data_directory
from link_bot_gazebo.gazebo_services import GazeboServices, restore_gazebo
from link_bot_pycommon.get_scenario import get_scenario
from merrrt_visualization.rviz_animation_controller import RvizSimpleStepper
from moonshine.filepath_tools import load_hjson
from moonshine.geometry import transform_dict_of_points_vectors
from std_msgs.msg import Float32

DEBUG_VIZ = True


def save_hparams(hparams, full_output_directory):
    hparams_filename = full_output_directory / 'params.hjson'
    with hparams_filename.open("w") as hparams_file:
        hjson.dump(hparams, hparams_file)


def sample_transform(rng, scaling):
    a = 0.25
    lower = np.array([-a, -a, -a, -np.pi, -np.pi, -np.pi])
    upper = np.array([a, a, a, np.pi, np.pi, np.pi])
    transform = rng.uniform(lower, upper).astype(np.float32) * scaling
    return transform
    # return [0, 0, 0, 0, 0, np.pi / 4]


@ros_init.with_ros("learn_invariance")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=pathlib.Path)
    parser.add_argument('outdir', type=pathlib.Path)
    parser.add_argument('--n-output-examples', type=int, default=1000)

    args = parser.parse_args()

    dataset_loader = NewDynamicsDatasetLoader([args.dataset])
    dataset = dataset_loader.get_dataset(mode='all').batch(batch_size=1)

    full_output_directory = data_directory(args.outdir)

    full_output_directory.mkdir(exist_ok=True)
    print(Fore.GREEN + full_output_directory.as_posix() + Fore.RESET)

    scenario_name = "floating_rope"
    hparams = {
        'made_from': args.dataset.as_posix(),
        'scenario':  scenario_name,
    }
    save_hparams(hparams, full_output_directory)

    gz = GazeboServices()
    gz.setup_env(verbose=0, real_time_rate=0, max_step_size=0.01)
    s = get_scenario(scenario_name)
    params = load_hjson(pathlib.Path("../link_bot_data/collect_dynamics_params/floating_rope.hjson"))
    s.on_before_get_state_or_execute_action()
    rng = np.random.RandomState(0)
    action_rng = np.random.RandomState(0)

    scaling_type = 'linear'

    stepper = RvizSimpleStepper()

    infinite_dataset = cycle(dataset)
    scaling = 1e-6
    for example_idx in progressbar(range(args.n_output_examples), widgets=base_dataset.widgets):
        if scaling_type == 'linear':
            scaling = example_idx / args.n_output_examples
        else:
            scaling = scaling * 1.01
            if scaling >= 1:
                return

        example = next(infinite_dataset)
        link_states_before = example['link_states'][0]  # t=0 arbitrarily
        restore_gazebo(gz, link_states_before, s)

        environment = s.get_environment(params)

        if example_idx % 100 == 0:
            print(scaling)
        transform = sample_transform(rng, scaling)  # uniformly sample an se3 transformation
        m = transformations.compose_matrix(angles=transform[3:], translate=transform[:3])

        state_before = s.get_state()
        action, _ = s.sample_action(action_rng,
                                    environment,
                                    state_before,
                                    params,
                                    validate=True)  # sample an action

        s.execute_action(environment, state_before, action)
        state_after = s.get_state()
        link_states_after = state_after['link_states']

        # set the simulator to the augmented before state
        link_states_before_aug = transform_link_states(m, link_states_before)
        link_states_after_aug = transform_link_states(m, link_states_after)
        state_points_keys = ['rope', 'left_gripper', 'right_gripper']
        action_points_keys = ['left_gripper_position', 'right_gripper_position']
        state_before_aug = transform_dict_of_points_vectors(m, state_before, state_points_keys)
        state_before_aug['link_states'] = link_states_before_aug
        state_after_aug_expected = transform_dict_of_points_vectors(m, state_after, state_points_keys)
        state_after_aug_expected['link_states'] = link_states_after_aug
        action_aug = transform_dict_of_points_vectors(m, action, action_points_keys)

        # set the simulator state to make the augmented state
        restore_gazebo(gz, link_states_before_aug, s)

        # execute the action and record the aug after state
        s.execute_action(environment, state_before_aug, action_aug)
        state_after_aug = s.get_state()

        if DEBUG_VIZ:
            s.plot_state_rviz(state_before, label='state_before', color='#ff0000')
            s.plot_action_rviz(state_before, action, label='action', color='#ff0000')
            s.plot_state_rviz(state_after, label='state_after', color='#aa0000')
            s.plot_state_rviz(state_before_aug, label='state_before_aug', color='#00ff00')
            s.plot_action_rviz(state_before_aug, action_aug, label='action_aug', color='#00ff00')
            s.plot_state_rviz(state_after_aug, label='state_after_aug', color='#00aa00')
            s.plot_state_rviz(state_after_aug_expected, label='state_after_aug_expected', color='#ffffff')
            error_viz = s.classifier_distance(state_after_aug, state_after_aug_expected)
            s.plot_error_rviz(error_viz)
            # stepper.step()

        out_example = {
            'state_before':             state_before,
            'state_after':              state_after,
            'action':                   action,
            'transformation':           transform,
            'state_before_aug':         state_before_aug,
            'state_after_aug':          state_after_aug,
            'state_after_aug_expected': state_after_aug_expected,
            'action_aug':               action_aug,
            'environment':              environment,
        }
        pkl_write_example(full_output_directory, out_example, example_idx)


if __name__ == '__main__':
    main()
