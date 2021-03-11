#!/usr/bin/env python
import argparse
import json
import pathlib

import colorama
import matplotlib.pyplot as plt
import numpy as np

import rospy
from link_bot_classifiers.analysis_utils import predict, execute
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.pycommon import make_dict_tf_float32
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import numpify
from state_space_dynamics import dynamics_utils

limit_gpu_mem(1)


def main():
    colorama.init(autoreset=True)
    plt.style.use("slides")
    np.set_printoptions(precision=3, suppress=True, linewidth=200)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("fwd_model_dir", help="load this saved forward model file", type=pathlib.Path, nargs='+')
    parser.add_argument("test_config", help="json file describing the test", type=pathlib.Path)
    parser.add_argument("labeling_params", help='labeling params', type=pathlib.Path)

    args = parser.parse_args()

    rospy.init_node('test_model_from_gazebo')

    test_config = json.load(args.test_config.open("r"))
    labeling_params = json.load(args.labeling_params.open("r"))
    labeling_state_key = labeling_params['state_key']

    # read actions from config
    actions = [numpify(a) for a in test_config['actions']]
    n_actions = len(actions)
    time_steps = np.arange(n_actions + 1)

    fwd_model, _ = dynamics_utils.load_generic_model(args.fwd_model_dir)

    service_provider = GazeboServices()
    service_provider.setup_env(verbose=0, real_time_rate=0,
                               max_step_size=fwd_model.data_collection_params['max_step_size'])
    environment = fwd_model.scenario.get_environment(params=fwd_model.data_collection_params)
    start_state = fwd_model.scenario.get_state()
    start_state = make_dict_tf_float32(start_state)
    start_states = [start_state]
    expanded_actions = [[actions]]
    predicted_states = predict(fwd_model, environment, start_states, expanded_actions, n_actions, 1, 1)

    scenario = fwd_model.scenario
    actual_states_lists = execute(service_provider, scenario, start_states, expanded_actions)

    visualize(
        scenario,
        environment,
        actual_states_lists,
        actions,
        predicted_states,
        labeling_state_key,
        time_steps
    )


def visualize(scenario, environment, actual_states_lists, actions, predicted_states, labeling_state_key, time_steps):
    for actual_states_list, predicted_states_list in zip(actual_states_lists, predicted_states):
        for actual_states, predicted_states in zip(actual_states_list, predicted_states_list):
            scenario.plot_environment_rviz(environment)

            anim = RvizAnimationController(time_steps)

            while not anim.done:
                t = anim.t()
                s_t = actual_states[t]
                s_t_pred = predicted_states[t]
                scenario.plot_state_rviz(s_t, label='actual', color='#ff0000aa')
                scenario.plot_state_rviz(s_t_pred, label='predicted', color='#0000ffaa')
                if t < len(actions):
                    action_t = actions[t]
                    scenario.plot_tree_action(s_t, action_t)
                else:
                    action_t = actions[-1]
                    scenario.plot_tree_action(s_t, action_t)

                if t > 0:
                    distance = np.linalg.norm(s_t[labeling_state_key] - s_t_pred[labeling_state_key])
                    print(f"t={t}, d={distance}")

                anim.step()


if __name__ == '__main__':
    main()
