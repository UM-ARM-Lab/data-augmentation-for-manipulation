from typing import Dict, Optional, List

import numpy as np
from matplotlib import cm

import rospy
from geometry_msgs.msg import Point
from link_bot_data.dataset_utils import add_predicted
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.pycommon import vector_to_points_2d
from moonshine.indexing import index_time_with_metadata, index_time, index_batch_time_with_metadata, index_batch_time, \
    index_state_action_with_metadata
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker


def plot_rope_configuration(ax, rope_configuration, linewidth=None, linestyle=None, s=1, label=None, scatt=True,
                            **kwargs):
    xs, ys = vector_to_points_2d(rope_configuration)
    if scatt:
        ax.scatter(xs, ys, s=s, **kwargs)
    return ax.plot(xs, ys, linewidth=linewidth, label=label, linestyle=linestyle, **kwargs)


def my_arrow(xs, ys, us, vs, scale=0.2):
    xs = np.array(xs)
    ys = np.array(ys)
    us = np.array(us)
    vs = np.array(vs)

    thetas = np.arctan2(vs, us)
    head_lengths = np.sqrt(np.square(us) + np.square(vs)) * scale
    theta1s = 3 * np.pi / 4 + thetas
    u1_primes = np.cos(theta1s) * head_lengths
    v1_primes = np.sin(theta1s) * head_lengths
    theta2s = thetas - 3 * np.pi / 4
    u2_primes = np.cos(theta2s) * head_lengths
    v2_primes = np.sin(theta2s) * head_lengths

    return ([xs, xs + us], [ys, ys + vs]), \
           ([xs + us, xs + us + u1_primes], [ys + vs, ys + vs + v1_primes]), \
           ([xs + us, xs + us + u2_primes], [ys + vs, ys + vs + v2_primes])


def plot_arrow(ax, xs, ys, us, vs, color, **kwargs):
    xys1, xys2, xys3 = my_arrow(xs, ys, us, vs)
    lines = []
    lines.append(ax.plot(xys1[0], xys1[1], c=color, **kwargs)[0])
    lines.append(ax.plot(xys2[0], xys2[1], c=color, **kwargs)[0])
    lines.append(ax.plot(xys3[0], xys3[1], c=color, **kwargs)[0])
    return lines


def update_arrow(lines, xs, ys, us, vs):
    xys1, xys2, xys3 = my_arrow(xs, ys, us, vs)
    lines[0].set_data(xys1[0], xys1[1])
    lines[1].set_data(xys2[0], xys2[1])
    lines[2].set_data(xys3[0], xys3[1])


def plot_extents(ax, extent, linewidth=6, **kwargs):
    line = ax.plot([extent[0], extent[1], extent[1], extent[0], extent[0]],
                   [extent[2], extent[2], extent[3], extent[3], extent[2]],
                   linewidth=linewidth,
                   **kwargs)[0]
    return line


def rviz_arrow(position: np.ndarray,
               target_position: np.ndarray,
               r: float,
               g: float,
               b: float,
               a: float,
               label: str = 'arrow',
               idx: int = 0,
               **kwargs):
    arrow = Marker()
    arrow.action = Marker.ADD  # create or modify
    arrow.type = Marker.ARROW
    arrow.header.frame_id = "world"
    arrow.header.stamp = rospy.Time.now()
    arrow.ns = label
    arrow.id = idx

    arrow.scale.x = 0.01
    arrow.scale.y = 0.02
    arrow.scale.z = 0

    arrow.pose.orientation.w = 1

    start = Point()
    start.x = position[0]
    start.y = position[1]
    start.z = position[2]
    end = Point()
    end.x = target_position[0]
    end.y = target_position[1]
    end.z = target_position[2]
    arrow.points.append(start)
    arrow.points.append(end)

    arrow.color.r = r
    arrow.color.g = g
    arrow.color.b = b
    arrow.color.a = a

    return arrow


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.figure()
    ax = plt.gca()
    r = 4
    for theta in np.linspace(-np.pi, np.pi, 36):
        u = np.cos(theta) * r
        v = np.sin(theta) * r
        plot_arrow(ax, 2, 0, u, v, 'r')
    plt.axis("equal")
    plt.show()


def recovery_transition_viz_t(metadata: Dict, state_keys: List[str]):
    def _recovery_transition_viz_t(scenario: ExperimentScenario, example: Dict, t: int):
        e_t = index_time_with_metadata(metadata, example, state_keys, t=t)
        scenario.plot_state_rviz(e_t, label='', color='#ff0000ff', scale=1.1)

    return _recovery_transition_viz_t


def classifier_transition_viz_t(metadata: Dict, predicted_state_keys, true_state_keys: Optional):
    def _classifier_transition_viz_t(scenario: ExperimentScenario, example: Dict, t: int):
        pred_t = index_time_with_metadata(metadata, example, predicted_state_keys, t=t)
        scenario.plot_state_rviz(pred_t, label='predicted', color='#0000ffff')

        label_t = example['is_close'][t]
        scenario.plot_is_close(label_t)

        if true_state_keys is not None:
            true_t = index_time_with_metadata(metadata, example, true_state_keys, t=t)
            scenario.plot_state_rviz(true_t, label='actual', color='#ff0000ff', scale=1.1)

    return _classifier_transition_viz_t


def viz_state_action_for_model_t(metadata: Dict, fwd_model: BaseDynamicsFunction):
    def _viz_state_action_t(scenario: ExperimentScenario, example: Dict, t: int):
        s_t = index_time_with_metadata(metadata, example, fwd_model.state_keys, t=t)
        action_s_t, a_t = index_state_action_with_metadata(metadata,
                                                           example,
                                                           fwd_model.state_keys,
                                                           fwd_model.action_keys, t=t)
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


def init_viz_action(metadata: Dict, action_keys, state_keys):
    def _init_viz_action(scenario: ExperimentScenario, example: Dict):
        action = {k: example[k][0] for k in action_keys}
        pred_0 = index_time_with_metadata(metadata, example, state_keys, t=0)
        scenario.plot_action_rviz(pred_0, action)

    return _init_viz_action


def viz_env_t(env_keys):
    def _viz_env_t(scenario: ExperimentScenario, example: Dict, t: Optional[int] = None):
        env_t = index_time(e=example, time_indexed_keys=env_keys, t=t, inclusive=True)
        scenario.plot_environment_rviz(env_t)

    return _viz_env_t


def init_viz_env(scenario: ExperimentScenario, example: Dict, t: Optional[int] = None):
    # the unused t arg makes it so we can pass this as either a t_func or a init_func
    scenario.plot_environment_rviz(example)


def stdev_viz_t(pub: rospy.Publisher):
    return float32_viz_t(pub, add_predicted('stdev'))


def recovery_probability_viz(pub: rospy.Publisher):
    return float32_viz(pub, 'recovery_probability')


def float32_viz(pub: rospy.Publisher, key: str):
    def _data_viz(scenario: ExperimentScenario, example: Dict):
        data_msg = Float32()
        data_msg.data = example[key][0]
        pub.publish(data_msg)

    return _data_viz


def float32_viz_t(pub: rospy.Publisher, key: str):
    def _data_viz_t(scenario: ExperimentScenario, example: Dict, t: int):
        data_t = example[key][t, 0]
        data_msg = Float32()
        data_msg.data = data_t
        pub.publish(data_msg)

    return _data_viz_t
