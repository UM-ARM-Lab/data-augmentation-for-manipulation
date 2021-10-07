from typing import Dict, Optional, List

import numpy as np
from matplotlib import colors

import rospy
from link_bot_data.dataset_utils import add_predicted
from link_bot_pycommon.matplotlib_utils import adjust_lightness
from link_bot_pycommon.pycommon import vector_to_points_2d
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.indexing import index_time_with_metadata, index_state_action_with_metadata
from moonshine.moonshine_utils import numpify
from std_msgs.msg import Float32


def plot_rope_configuration(ax, rope_configuration, linewidth=None, linestyle=None, s=1, label=None, scatt=True,
                            **kwargs):
    xs, ys = vector_to_points_2d(rope_configuration)
    if scatt:
        ax.scatter(xs, ys, s=s, **kwargs)
    return ax.plot(xs, ys, linewidth=linewidth, label=label, linestyle=linestyle, **kwargs)


def plot_extents(ax, extent, linewidth=6, **kwargs):
    line = ax.plot([extent[0], extent[1], extent[1], extent[0], extent[0]],
                   [extent[2], extent[2], extent[3], extent[3], extent[2]],
                   linewidth=linewidth,
                   **kwargs)[0]
    return line


def dynamics_viz_t(metadata: Dict, state_metadata_keys, state_keys, action_keys):
    def _dynamics_transition_viz_t(scenario: ScenarioWithVisualization, example: Dict, t: int):
        s_t = index_time_with_metadata(metadata, example, state_metadata_keys + state_keys, t=t)
        try_adding_aco(state=s_t, example=example)
        scenario.plot_state_rviz(s_t, label='actual', color='#ff0000ff')

        s_for_a_t, a_t = index_state_action_with_metadata(example,
                                                          state_keys=state_keys,
                                                          state_metadata_keys=state_metadata_keys,
                                                          action_keys=action_keys,
                                                          t=t)
        scenario.plot_action_rviz(s_for_a_t, a_t)

    return _dynamics_transition_viz_t


def recovery_transition_viz_t(metadata: Dict, state_keys: List[str]):
    def _recovery_transition_viz_t(scenario: ScenarioWithVisualization, example: Dict, t: int):
        e_t = index_time_with_metadata(metadata, example, state_keys, t=t)
        scenario.plot_state_rviz(e_t, label='', color='#ff0000ff', scale=1.1)

    return _recovery_transition_viz_t


def classifier_transition_viz_t(metadata: Dict, state_metadata_keys, predicted_state_keys, true_state_keys: Optional):
    def _classifier_transition_viz_t(scenario: ScenarioWithVisualization, example: Dict, t: int):
        pred_t = index_time_with_metadata(metadata, example, state_metadata_keys + predicted_state_keys, t=t)
        print(example['predicted/accept_probability'])
        try_adding_aco(state=pred_t, example=example)
        scenario.plot_state_rviz(pred_t, label='predicted', color='#0000ffff')

        label_t = example['is_close'][t]
        scenario.plot_is_close(label_t)

        if true_state_keys is not None:
            true_t = index_time_with_metadata(metadata, example, state_metadata_keys + true_state_keys, t=t)
            try_adding_aco(state=true_t, example=example)
            scenario.plot_state_rviz(true_t, label='actual', color='#ff0000ff', scale=1.1)

        if 'error' in example:
            scenario.plot_error_rviz(example['error'][t])

    return _classifier_transition_viz_t


def init_viz_action(metadata: Dict, action_keys, state_keys):
    def _init_viz_action(scenario: ScenarioWithVisualization, example: Dict):
        action = {k: example[k][0] for k in action_keys}
        pred_0 = index_time_with_metadata(metadata, example, state_keys, t=0)
        scenario.plot_action_rviz(pred_0, action)

    return _init_viz_action


def init_viz_env(scenario: ScenarioWithVisualization, example: Dict, t: Optional[int] = None):
    # the unused t arg makes it so we can pass this as either a t_func or a init_func
    scenario.plot_environment_rviz(example)


def stdev_viz_t(pub: rospy.Publisher):
    return float32_viz_t(pub, add_predicted('stdev'))


def recovery_probability_viz(pub: rospy.Publisher):
    return float32_viz(pub, 'recovery_probability')


def float32_viz(pub: rospy.Publisher, key: str):
    def _data_viz(scenario: ScenarioWithVisualization, example: Dict):
        data_msg = Float32()
        data_msg.data = example[key][0]
        pub.publish(data_msg)

    return _data_viz


def float32_viz_t(pub: rospy.Publisher, key: str):
    def _data_viz_t(scenario: ScenarioWithVisualization, example: Dict, t: int):
        data_t = example[key][t, 0]
        data_msg = Float32()
        data_msg.data = data_t
        pub.publish(data_msg)

    return _data_viz_t


def color_violinplot(parts, color):
    r, g, b, a = colors.to_rgba(color)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(a)
    for partname in ['cmeans', ]:
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor('#dddddd')
            vp.set_alpha(a)
            vp.set_linewidth(3)
    for partname in ['cbars', 'cmins', 'cmaxes']:
        color_dark = adjust_lightness(color, 0.1)
        vp = parts[partname]
        vp.set_edgecolor(color_dark)
        vp.set_linewidth(1)
        vp.set_alpha(a)


def noise_x_like(y, nominal_x, noise=0.01):
    return np.random.normal(nominal_x, noise, size=y.shape[0])


def noisey_1d_scatter(ax, x, position, noise=0.01, **kwargs):
    ax.scatter(noise_x_like(x, position, noise), x, **kwargs)


def try_adding_aco(state: Dict, example: Dict):
    try:
        state['attached_collision_objects'] = example['scene_msg'].robot_state.attached_collision_objects
    except Exception:
        pass


def plot_classifier_state_b_t(scenario, state_keys, input_dict, b, t, label: str, color='red'):
    state_t = numpify({k: input_dict[add_predicted(k)][b, t] for k in state_keys})
    state_t['joint_names'] = input_dict['joint_names'][b, t]
    scenario.plot_state_rviz(state_t, label=label, color=color)

    if 'is_close' in input_dict:
        scenario.plot_is_close(input_dict['is_close'][b, 1])
    else:
        scenario.plot_is_close(None)

    if 'error' in input_dict:
        error_t = input_dict['error'][b, 1]
        scenario.plot_error_rviz(error_t)
    else:
        scenario.plot_error_rviz(-999)


def plot_classifier_state_t(scenario, state_keys, input_dict, t, label: str, color='red'):
    state_t = numpify({k: input_dict[add_predicted(k)][t] for k in state_keys})
    state_t['joint_names'] = input_dict['joint_names'][t]
    scenario.plot_state_rviz(state_t, label=label, color=color)

    if 'is_close' in input_dict:
        scenario.plot_is_close(input_dict['is_close'][1])
    else:
        scenario.plot_is_close(None)

    if 'error' in input_dict:
        error_t = input_dict['error'][1]
        scenario.plot_error_rviz(error_t)
    else:
        scenario.plot_error_rviz(-999)
