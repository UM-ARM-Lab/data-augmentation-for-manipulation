from typing import Dict

from matplotlib import cm

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.indexing import index_time_with_metadata, index_state_action_with_metadata, index_time, index_batch_time, \
    index_batch_time_with_metadata
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


def viz_state_action_for_model_t(metadata: Dict, fwd_model: BaseDynamicsFunction):
    def _viz_state_action_t(scenario: ExperimentScenario, example: Dict, t: int):
        s_t = index_time_with_metadata(metadata, example, fwd_model.state_keys, t=t)
        action_s_t, a_t = index_state_action_with_metadata(example,
                                                           state_keys=fwd_model.state_keys,
                                                           state_metadata_keys=fwd_model.state_metadata_keys,
                                                           action_keys=fwd_model.action_keys,
                                                           t=t,
                                                           metadata=metadata)
        scenario.plot_state_rviz(s_t, label='', color='#ff0000ff')
        scenario.plot_action_rviz(action_s_t, a_t, label='')

    return _viz_state_action_t


def viz_transition_for_model_t(metadata: Dict, fwd_model: BaseDynamicsFunction):
    def _viz_transition_t(scenario: ExperimentScenario, example: Dict, t: int):
        action = index_time(example, fwd_model.action_keys, t=t, inclusive=False)
        s0 = index_time_with_metadata(metadata, example, fwd_model.state_keys, t=0)
        s1 = index_time_with_metadata(metadata, example, fwd_model.state_keys, t=1)
        if 'accept_probablity' in example:
            accept_probability_t = example['accept_probability'][t]
            color = cm.Reds(accept_probability_t)
        else:
            color = "#aa2222aa"
        scenario.plot_state_rviz(s0, label='', color='#ff0000ff')
        scenario.plot_state_rviz(s1, label='predicted', color=color)
        scenario.plot_action_rviz(s0, action, label='')

    return _viz_transition_t


def viz_transition_for_model_t_batched(metadata: Dict, fwd_model: BaseDynamicsFunction):
    def _viz_transition_t(scenario: ExperimentScenario, example: Dict, t: int):
        action = index_batch_time(example, fwd_model.action_keys, b=t, t=0)
        s0 = index_batch_time_with_metadata(metadata, example, fwd_model.state_keys, b=t, t=0)
        s1 = index_batch_time_with_metadata(metadata, example, fwd_model.state_keys, b=t, t=1)
        if 'accept_probablity' in example:
            accept_probability_t = example['accept_probability'][t]
            color = cm.Reds(accept_probability_t)
        else:
            color = "#aa2222aa"
        scenario.plot_state_rviz(s0, label='', color='#ff0000ff')
        scenario.plot_state_rviz(s1, label='predicted', color=color)
        scenario.plot_action_rviz(s0, action, label='')

    return _viz_transition_t


def init_viz_action_for_model(metadata: Dict, fwd_model: BaseDynamicsFunction):
    def _init_viz_action(scenario: ExperimentScenario, example: Dict):
        action = {k: example[k][0] for k in fwd_model.action_keys}
        pred_0 = index_time_with_metadata(metadata, example, fwd_model.state_keys, t=0)
        scenario.plot_action_rviz(pred_0, action)

    return _init_viz_action
