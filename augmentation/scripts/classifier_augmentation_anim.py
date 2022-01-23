#!/usr/bin/env python
import argparse
import logging
import pathlib
from copy import deepcopy
from datetime import datetime
from time import sleep

import pyautogui as pyautogui
import tensorflow as tf

from arc_utilities import ros_init
from arc_utilities.algorithms import nested_dict_update
from augmentation.augment_dataset import make_aug_opt
from link_bot_data.dataset_utils import add_predicted
from link_bot_data.new_classifier_dataset import NewClassifierDatasetLoader
from link_bot_data.visualization import classifier_transition_viz_t
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_hjson
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import remove_batch
from tf.transformations import quaternion_from_euler

limit_gpu_mem(None)


@ros_init.with_ros("classifier_augmentation_anim")
def main():
    tf.get_logger().setLevel(logging.FATAL)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('in_idx', type=int)
    parser.add_argument('aug_seed', type=int)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--hparams', type=pathlib.Path, default=pathlib.Path("aug_hparams/rope.hjson"))

    args = parser.parse_args()

    dataset_dir = args.dataset_dir

    scenario = get_scenario("dual_arm_rope_sim_val_with_robot_feasibility_checking")
    common_hparams = load_hjson(pathlib.Path("aug_hparams/common.hjson"))
    hparams = load_hjson(args.hparams)
    hparams = nested_dict_update(common_hparams, hparams)
    hparams['seed'] = args.aug_seed

    dataset_loader = NewClassifierDatasetLoader([dataset_dir])
    debug_state_keys = [add_predicted(k) for k in dataset_loader.state_keys]

    date_str = datetime.now().strftime('%H-%M-%S')
    outdir = pathlib.Path('anims') / f"{args.dataset_dir.name}_{date_str}"
    outdir.mkdir(exist_ok=True, parents=True)

    region = (800, 100, 900, 900)

    def post_init_cb():
        sleep(0.5)
        pyautogui.screenshot(outdir / f"post_init.png", region=region)

    def post_step_cb(i):
        sleep(0.5)
        pyautogui.screenshot(outdir / f"post_step_{i}.png", region=region)

    def post_project_cb(i):
        sleep(0.5)
        pyautogui.screenshot(outdir / f"post_project_{i}.png", region=region)

    aug = make_aug_opt(scenario, dataset_loader, hparams, debug_state_keys, 1,
                       post_init_cb, post_step_cb, post_project_cb)

    dataset = dataset_loader.get_datasets('all').skip(args.in_idx).take(1).batch(1)

    scenario.tf.send_transform([-1, 0, 1], quaternion_from_euler(0, 0, 0), 'hdt_michigan_root', 'anim_camera')

    # plot the environment, rope at t=0, and rope at t=1
    viz_f = classifier_transition_viz_t(metadata={},
                                        state_metadata_keys=dataset_loader.state_metadata_keys,
                                        predicted_state_keys=dataset_loader.predicted_state_keys,
                                        true_state_keys=None)

    original = next(iter(dataset))
    original_no_batch = deepcopy(remove_batch(original))

    scenario.reset_viz()
    for _ in range(3):
        scenario.plot_environment_rviz(original_no_batch)
        viz_f(scenario, original_no_batch, t=0, label='0')
        viz_f(scenario, original_no_batch, t=1, label='1')

    sleep(0.5)
    pyautogui.screenshot(outdir / "original.png", region=region)

    scenario.reset_viz()

    time = original['time_idx'].shape[1]
    output = aug.aug_opt(original, batch_size=1, time=time)
    output = remove_batch(output)
    aug.delete_state_action_markers()

    scenario.reset_viz()
    for _ in range(3):
        scenario.plot_environment_rviz(output)
        viz_f(scenario, output, t=0, label='0')
        viz_f(scenario, output, t=1, label='1')

    sleep(0.5)
    pyautogui.screenshot(outdir / "output.png", region=region)


if __name__ == '__main__':
    main()
