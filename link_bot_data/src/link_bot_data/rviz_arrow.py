import numpy as np

import rospy
from geometry_msgs.msg import Point
from link_bot_data.color_from_kwargs import color_from_kwargs
from visualization_msgs.msg import Marker


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

    arrow.scale.x = kwargs.get('scale', 1.0) * 0.0025
    arrow.scale.y = kwargs.get('scale', 1.0) * 0.004
    arrow.scale.z = kwargs.get('scale', 1.0) * 0.006

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
