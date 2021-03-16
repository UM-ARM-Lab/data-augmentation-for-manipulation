from typing import Dict, Optional, List

import numpy as np
from matplotlib import colors

import rospy
from geometry_msgs.msg import Point
from link_bot_data.dataset_utils import add_predicted
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.matplotlib_utils import adjust_lightness
from link_bot_pycommon.pycommon import vector_to_points_2d
from moonshine.indexing import index_time_with_metadata
from std_msgs.msg import Float32, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


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


def color_from_kwargs(kwargs, r, g, b, a=1.0):
    """

    Args:
        kwargs:
        r:  default red
        g:  default green
        b:  default blue
        a:  refault alpha

    Returns:

    """
    if 'color' in kwargs:
        return ColorRGBA(*colors.to_rgba(kwargs["color"]))
    else:
        r = float(kwargs.get("r", r))
        g = float(kwargs.get("g", g))
        b = float(kwargs.get("b", b))
        a = float(kwargs.get("a", a))
        return ColorRGBA(r, g, b, a)


def rviz_arrow(position: np.ndarray,
               target_position: np.ndarray,
               label: str = 'arrow',
               **kwargs):
    idx = kwargs.get("idx", 0)
    color = color_from_kwargs(kwargs, 0, 0, 1.0)

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

    arrow.color = color

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


def classifier_transition_viz_t(metadata: Dict, state_metadata_keys, predicted_state_keys, true_state_keys: Optional):
    def _classifier_transition_viz_t(scenario: ExperimentScenario, example: Dict, t: int):
        pred_t = index_time_with_metadata(metadata, example, state_metadata_keys + predicted_state_keys, t=t)
        scenario.plot_state_rviz(pred_t, label='predicted', color='#0000ffff')

        label_t = example['is_close'][t]
        scenario.plot_is_close(label_t)

        if true_state_keys is not None:
            true_t = index_time_with_metadata(metadata, example, state_metadata_keys + true_state_keys, t=t)
            scenario.plot_state_rviz(true_t, label='actual', color='#ff0000ff', scale=1.1)

    return _classifier_transition_viz_t


def init_viz_action(metadata: Dict, action_keys, state_keys):
    def _init_viz_action(scenario: ExperimentScenario, example: Dict):
        action = {k: example[k][0] for k in action_keys}
        pred_0 = index_time_with_metadata(metadata, example, state_keys, t=0)
        scenario.plot_action_rviz(pred_0, action)

    return _init_viz_action


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


def make_delete_marker(marker_id: int, ns: str):
    m = Marker(action=Marker.DELETE, ns=ns, id=marker_id)
    msg = MarkerArray(markers=[m])
    return msg


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
