#!/usr/bin/env python
import argparse
import pathlib

import hjson
import matplotlib.pyplot as plt
import numpy as np

from arc_utilities import ros_init
from link_bot_classifiers.classifier_analysis_utils import predict
from link_bot_planning.plan_and_execute import execute_actions
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.gpu_config import limit_gpu_mem
from moonshine.numpify import numpify
from state_space_dynamics.train_test_dynamics import load_udnn_model_wrapper

limit_gpu_mem(None)


@ros_init.with_ros("test_model_from_gazebo")
def main():
    plt.style.use("slides")
    np.set_printoptions(precision=3, suppress=True, linewidth=200)

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="load this saved forward model file")
    parser.add_argument("test_config", help="hjson file describing the test", type=pathlib.Path)

    args = parser.parse_args()

    test_config = hjson.load(args.test_config.open("r"))

    # read actions from config
    actions = [numpify(a) for a in test_config]
    n_actions = len(actions)
    time_steps = np.arange(n_actions + 1)

    model = load_udnn_model_wrapper(args.checkpoint)

    scenario = model.scenario

    start_state = scenario.get_state()
    environment = scenario.get_environment()

    predicted_states = predict(model, environment, start_state, actions)

    execution_result = execute_actions(scenario, environment, start_state, actions)
    actual_states = execution_result.path

    visualize(scenario, environment, actual_states, actions, predicted_states)


def predict(model, environment, start_state, actions):
    predicted_states = []
    input_dict = {}
    input_dict
    model(input_dict)
    return predicted_states


def visualize(scenario, environment, actual_states, actions, predicted_states):
    scenario.plot_environment_rviz(environment)

    anim = RvizAnimationController(n_time_steps=len(actual_states))

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
            distance = scenario.classifier_distance(s_t, s_t_pred)
            print(f"t={t}, d={distance}")

        anim.step()


if __name__ == '__main__':
    main()
