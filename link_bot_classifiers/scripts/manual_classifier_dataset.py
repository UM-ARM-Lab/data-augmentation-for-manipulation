#!/usr/bin/env python
import argparse
import pathlib

import colorama
import numpy as np

import rospy
from arc_utilities import ros_init
from link_bot_data.classifier_dataset_utils import add_model_error
from link_bot_data.dataset_utils import tf_write_example, add_predicted, use_gt_rope
from link_bot_data.files_dataset import FilesDataset
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.pycommon import try_make_dict_tf_float32
from link_bot_pycommon.serialization import my_hdump
from moonshine.filepath_tools import load_hjson
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_tensors
from state_space_dynamics import dynamics_utils


@ros_init.with_ros("manual_classifier_dataset")
def main():
    colorama.init(autoreset=True)
    np.set_printoptions(precision=3, suppress=True, linewidth=200)

    parser = argparse.ArgumentParser()
    parser.add_argument("fwd_model_dir", help="fwd model dirs", type=pathlib.Path, nargs="+")
    parser.add_argument("labeling_params", type=pathlib.Path)
    parser.add_argument("data_collection_params", type=pathlib.Path, help='use one of the hjson files for phase2')
    parser.add_argument("outdir", help="output directory", type=pathlib.Path)

    args = parser.parse_args()

    scenario = get_scenario("dual_arm_rope_sim_val")

    fwd_model, _ = dynamics_utils.load_generic_model(args.fwd_model_dir, scenario)

    files = FilesDataset(args.outdir)
    labeling_params = load_hjson(args.labeling_params)
    params = load_hjson(args.data_collection_params)
    example_idx = 0

    args.outdir.mkdir(exist_ok=True)

    new_hparams_filename = args.outdir / 'hparams.hjson'
    classifier_dataset_hparams = params
    classifier_dataset_hparams['scenario'] = scenario.simple_name()
    classifier_dataset_hparams['labeling_params'] = labeling_params
    classifier_dataset_hparams['fwd_model_hparams'] = fwd_model.hparams
    classifier_dataset_hparams['predicted_state_keys'] = fwd_model.state_keys
    classifier_dataset_hparams['dataset_dir'] = 'manual'
    classifier_dataset_hparams['true_state_keys'] = fwd_model.state_keys + ['gt_rope']
    classifier_dataset_hparams['state_metadata_keys'] = fwd_model.state_metadata_keys
    classifier_dataset_hparams['action_keys'] = fwd_model.action_keys

    input("press enter to begin")

    has_written_hparams = False
    while not rospy.is_shutdown():
        environment = scenario.get_environment(params)
        before_state = scenario.get_state()

        # let the user move the robot
        k = input("press enter to record the transition")
        if k == 'q':
            print("Exiting")
            break

        after_state = scenario.get_state()

        states = sequence_of_dicts_to_dict_of_tensors([before_state, after_state])
        action = {
            'left_gripper_position':  after_state['left_gripper'],
            'right_gripper_position': after_state['right_gripper'],
        }
        actions_list = [action]

        if not has_written_hparams:
            has_written_hparams = True
            classifier_dataset_hparams['env_keys'] = list(environment.keys())
            my_hdump(classifier_dataset_hparams, new_hparams_filename.open("w"), indent=2)

        # run prediction
        predictions, _ = fwd_model.propagate(environment, before_state, actions_list)

        # create example
        predictions = sequence_of_dicts_to_dict_of_tensors(predictions)
        actions = sequence_of_dicts_to_dict_of_tensors(actions_list)
        example = {
            'classifier_start_t': 0,
            'classifier_end_t':   2,
            'prediction_start_t': 0,
            'traj_idx':           example_idx,
            'time_idx':           [0, 1],
        }
        example.update(environment)
        example.update(states)
        example.update({add_predicted(k): v for k, v in predictions.items()})
        example.update(actions)
        example = use_gt_rope(example)
        add_model_error(states, labeling_params, example, predictions, scenario)

        # write example to file
        example = try_make_dict_tf_float32(example)
        full_filename = tf_write_example(args.outdir, example, example_idx)
        files.add(full_filename)
        example_idx += 1

    files.split()


if __name__ == '__main__':
    main()
