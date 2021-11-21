#!/usr/bin/env python
import argparse
import pathlib

import hjson
import numpy as np
import transformations
from colorama import Fore
from tqdm import trange

from arc_utilities import ros_init
from learn_invariance.transform_link_states import transform_link_states
from link_bot_data.dataset_utils import pkl_write_example, make_unique_outdir
from link_bot_gazebo.gazebo_services import restore_gazebo
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_hjson
from moonshine.geometry import transform_dict_of_points_vectors


def save_hparams(hparams, full_output_directory):
    hparams_filename = full_output_directory / 'params.hjson'
    with hparams_filename.open("w") as hparams_file:
        hjson.dump(hparams, hparams_file)


def scaling_gen(n, scaling_type='linear'):
    scaling = 1e-6
    for j in n:
        if scaling_type == 'linear':
            scaling = j / n
            yield scaling
        else:
            scaling = scaling * 1.01
            if scaling >= 1:
                return


@ros_init.with_ros("generate_invariance_dataset")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_collection_params', type=pathlib.Path)
    parser.add_argument('outdir', type=pathlib.Path)
    parser.add_argument('--n-test-states', type=int, default=5)
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    full_output_directory = make_unique_outdir(args.outdir)

    full_output_directory.mkdir(exist_ok=True)
    print(Fore.GREEN + full_output_directory.as_posix() + Fore.RESET)

    transformation_dim = 6
    n_output_examples = np.sqrt(10 ** transformation_dim)
    scenario_name = "dual_arm_rope_sim_val_with_robot_feasibility_checking"
    params = load_hjson(args.data_collection_params)

    hparams = {
        'scenario':               scenario_name,
        'data_collection_params': params,
        'transformation_dim':     transformation_dim,
        'n_output_examples':      n_output_examples,
    }
    save_hparams(hparams, full_output_directory)

    s = get_scenario(scenario_name)
    s.on_before_get_state_or_execute_action()

    transform_sampling_rng = np.random.RandomState(0)
    state_rng = np.random.RandomState(0)
    action_rng = np.random.RandomState(0)

    environment = s.get_environment(params)

    example_idx = 0
    for i in trange(args.n_test_states, position=1):
        s.tinv_move_to_random_state(state_rng)

        for scaling in trange(scaling_gen(n_output_examples), position=2):
            state_before = s.get_state()

            # NOTE: in general maybe we need a whole trajectory? it depends on the data right?
            action, _ = s.sample_action(action_rng,
                                        environment,
                                        state_before,
                                        params,
                                        validate=True)  # sample an action

            s.execute_action(environment, state_before, action)
            state_after = s.get_state()
            link_states_after = state_after['link_states']

            # sample a transformation
            transform = s.tinv_sample_transform(transform_sampling_rng, scaling)  # uniformly sample a transformation

            # set the simulator to the augmented before state
            state_before_aug = s.tinv_apply_transformation(

            )
            # m = transformations.compose_matrix(angles=transform[3:], translate=transform[:3])
            # link_states_before_aug = transform_link_states(m, link_states_before)
            # link_states_after_aug = transform_link_states(m, link_states_after)
            # state_points_keys = ['rope', 'left_gripper', 'right_gripper']
            # action_points_keys = ['left_gripper_position', 'right_gripper_position']
            # state_before_aug = transform_dict_of_points_vectors(m, state_before, state_points_keys)
            # state_before_aug['link_states'] = link_states_before_aug
            # state_after_aug_expected = transform_dict_of_points_vectors(m, state_after, state_points_keys)
            # state_after_aug_expected['link_states'] = link_states_after_aug
            # action_aug = transform_dict_of_points_vectors(m, action, action_points_keys)
            #
            # # set the simulator state to make the augmented state
            # restore_gazebo(gz, link_states_before_aug, s)

            # execute the action and record the aug after state
            s.execute_action(environment, state_before_aug, action_aug)

            state_after_aug = s.get_state()

            if args.visualize:
                s.plot_state_rviz(state_before, label='state_before', color='#ff0000')
                s.plot_action_rviz(state_before, action, label='action', color='#ff0000')
                s.plot_state_rviz(state_after, label='state_after', color='#aa0000')
                s.plot_state_rviz(state_before_aug, label='state_before_aug', color='#00ff00')
                s.plot_action_rviz(state_before_aug, action_aug, label='action_aug', color='#00ff00')
                s.plot_state_rviz(state_after_aug, label='state_after_aug', color='#00aa00')
                s.plot_state_rviz(state_after_aug_expected, label='state_after_aug_expected', color='#ffffff')
                error_viz = s.classifier_distance(state_after_aug, state_after_aug_expected)
                s.plot_error_rviz(error_viz)

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
            example_idx += 1


if __name__ == '__main__':
    main()
