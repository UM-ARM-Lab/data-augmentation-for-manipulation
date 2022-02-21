#!/usr/bin/env python
import pathlib
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import tensorflow as tf
from colorama import Fore
from matplotlib import cm
from numpy.random import RandomState

from link_bot_classifiers import nn_recovery_model2, nn_recovery_model
from link_bot_classifiers.base_recovery_policy import BaseRecoveryPolicy
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.pycommon import make_dict_tf_float32, log_scale_0_to_1
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.ensemble import Ensemble2
from moonshine.torch_and_tf_utils import add_batch

DEBUG_VIZ = False
POLICY_DEBUG_VIZ = False


@dataclass
class RecoveryDebugVizInfo:
    actions: List[Dict]
    recovery_probabilities: List
    states: List[Dict]
    environment: Dict

    def __len__(self):
        return len(self.states)


def shrink_extent(extent, d):
    extent = tf.reshape(extent, [3, 2])
    new_extent = tf.stack([extent[:, 0] + (extent[:, 0] * 0 + d), extent[:, 1] - (extent[:, 1] * 0 + d)], axis=1)
    return tf.reshape(new_extent, [-1])


class NNRecoveryPolicy(BaseRecoveryPolicy):

    def __init__(self, path: pathlib.Path, scenario: ExperimentScenario, rng: RandomState, u: Dict):
        super().__init__(path, scenario, rng, u)

        self.model = self.model_class()(hparams=self.params, batch_size=1, scenario=self.scenario)
        self.ckpt = tf.train.Checkpoint(model=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, path, max_to_keep=1)

        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(self.manager.latest_checkpoint) + Fore.RESET)
        else:
            raise RuntimeError("Failed to restore!!!")

        self.action_rng = RandomState(0)
        dataset_params = self.params['recovery_dataset_hparams']
        self.data_collection_params = dataset_params['data_collection_params']
        self.n_action_samples = dataset_params['labeling_params']['n_action_samples']

        self.noise_rng = RandomState(0)

    def from_example(self, inputs: Dict):
        inputs = self.model.preprocess_no_gradient(inputs, training=False)
        return self.model(inputs)

    def __call__(self, environment: Dict, state: Dict):
        # sample a bunch of actions (batched?) and pick the best one
        max_unstuck_probability = -1
        best_action = None

        action_params = deepcopy(self.data_collection_params)
        action_params['extent'] = shrink_extent(environment['extent'], 0.05).numpy()
        action_params['left_gripper_action_sample_extent'] = shrink_extent(environment['extent'], 0.05).numpy()
        action_params['right_gripper_action_sample_extent'] = shrink_extent(environment['extent'], 0.05).numpy()

        info = RecoveryDebugVizInfo(actions=[],
                                    states=[],
                                    recovery_probabilities=[],
                                    environment=environment)

        for _ in range(self.n_action_samples):
            self.scenario.last_action = None
            action, _ = self.scenario.sample_action(action_rng=self.action_rng,
                                                    environment=environment,
                                                    state=state,
                                                    action_params=action_params,
                                                    validate=False)  # not checking here since we check after adding noise
            action = self.scenario.add_action_noise(action, self.noise_rng)
            valid = self.scenario.is_action_valid(environment, state, action, action_params)
            if not valid:
                continue

            recovery_probability = self.compute_recovery_probability(environment, state, action)

            info.states.append(state)
            info.actions.append(action)
            info.recovery_probabilities.append(recovery_probability)

            if recovery_probability > max_unstuck_probability:
                max_unstuck_probability = recovery_probability
                best_action = action

        if POLICY_DEBUG_VIZ:
            self.debug_viz(info)

        return best_action

    def compute_recovery_probability(self, environment, state, action):
        recovery_model_inputs = {}
        recovery_model_inputs.update(environment)
        recovery_model_inputs.update(add_batch(state))  # add time dimension to state and action
        recovery_model_inputs.update(add_batch(action))
        recovery_model_inputs = add_batch(recovery_model_inputs)
        if 'scene_msg' in environment:
            recovery_model_inputs.pop('scene_msg')
        recovery_model_inputs = make_dict_tf_float32(recovery_model_inputs)
        recovery_model_inputs.update({
            'batch_size': 1,
            'time':       2,
        })
        recovery_model_inputs = self.model.preprocess_no_gradient(recovery_model_inputs, training=False)
        recovery_model_outputs = self.model(recovery_model_inputs, training=False)
        recovery_probability = recovery_model_outputs['probabilities']
        return recovery_probability

    def debug_viz(self, info: RecoveryDebugVizInfo):
        anim = RvizAnimationController(np.arange(len(info)))
        debug_viz_max_unstuck_probability = -1
        while not anim.done:
            i = anim.t()
            s_i = info.states[i]
            a_i = info.actions[i]
            p_i = info.recovery_probabilities[i]

            self.scenario.plot_recovery_probability(p_i)
            color_factor = log_scale_0_to_1(tf.squeeze(p_i), k=500)
            self.scenario.plot_action_rviz(s_i, a_i, label='proposed', color=cm.Greens(color_factor), idx=1)
            self.scenario.plot_environment_rviz(info.environment)
            self.scenario.plot_state_rviz(s_i, label='stuck_state')

            if p_i > debug_viz_max_unstuck_probability:
                debug_viz_max_unstuck_probability = p_i
                self.scenario.plot_action_rviz(s_i, a_i, label='best_proposed', color='g', idx=2)

            anim.step()

    def model_class(self):
        return nn_recovery_model.NNRecoveryModel


class NNRecoveryPolicy2(NNRecoveryPolicy):

    def model_class(self):
        return nn_recovery_model2.NNRecoveryModel


class NNRecoveryEnsemble(BaseRecoveryPolicy):
    def __init__(self, path, elements, constants_keys: List[str], rng: RandomState, u: Dict):
        self.ensemble = Ensemble2(elements, constants_keys)
        m0 = self.ensemble.elements[0]
        self.element_class = m0.__class__

        super().__init__(path, m0.scenario, rng, u)

    def from_example(self, example: Dict):
        mean, stdev = self.ensemble(self.element_class.propagate_from_example, example)
        return mean, stdev

    def __call__(self, environment: Dict, state: Dict):
        mean, stdev = self.ensemble(self.element_class.__call__, environment, state)
        return mean, stdev
