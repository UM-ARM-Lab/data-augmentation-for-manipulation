#!/usr/bin/env python
import pathlib
from typing import Optional, Callable

import numpy as np

from link_bot_classifiers import recovery_policy_utils, nn_recovery_policy
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_pycommon.get_scenario import get_scenario
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.torch_and_tf_utils import repeat


def test_recovery(recovery_model_dir: pathlib.Path,
                  n_actions: int,
                  saved_state: Optional[pathlib.Path],
                  generate_actions: Callable):
    scenario = get_scenario("dual_arm_rope_sim_val_with_robot_feasibility_checking")
    scenario.on_before_get_state_or_execute_action()

    rng = np.random.RandomState(0)
    recovery = recovery_policy_utils.load_generic_model(recovery_model_dir, scenario, rng)

    service_provider = GazeboServices()
    service_provider.setup_env(verbose=0,
                               real_time_rate=0,
                               max_step_size=0.01,
                               play=False)
    if saved_state:
        service_provider.restore_from_bag(saved_state)

    params = recovery.data_collection_params
    environment = scenario.get_environment(params)
    start_state = scenario.get_state()
    start_state_tiled = repeat(start_state, n_actions, axis=0, new_axis=True)

    anim = RvizAnimationController(n_time_steps=n_actions)
    actions = list(generate_actions(environment, start_state_tiled, scenario, params, n_actions))

    nn_recovery_policy.POLICY_DEBUG_VIZ = True
    recovery(environment, start_state)

    recovery_probabilities = []
    for action in actions:
        recovery_probability = recovery.compute_recovery_probability(environment, start_state, action)
        recovery_probabilities.append(recovery_probability)

    while not anim.done:
        t = anim.t()

        action = actions[t]
        recovery_probability = recovery_probabilities[t]

        scenario.plot_environment_rviz(environment)
        scenario.plot_state_rviz(start_state)
        scenario.plot_action_rviz(start_state, action)
        scenario.plot_recovery_probability(recovery_probability)

        anim.step()
