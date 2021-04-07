import colorsys
from typing import Iterable

import matplotlib.colors as mc
import numpy as np
from matplotlib import cm, colors
from matplotlib.figure import figaspect

from std_msgs.msg import ColorRGBA


def save_unconstrained_layout(fig, filename, dpi=250):
    fig.set_constrained_layout(False)
    fig.savefig(filename, bbox_inches='tight', dpi=dpi, transparent=True)


def state_image_to_cmap(state_image: np.ndarray, cmap=cm.viridis, binary_threshold=0.1):
    h, w, n_channels = state_image.shape
    new_image = np.zeros([h, w, 3])
    for channel_idx in range(n_channels):
        channel = np.take(state_image, indices=channel_idx, axis=-1)
        color = cmap(channel_idx / n_channels)[:3]
        rows, cols = np.where(channel > binary_threshold)
        new_image[rows, cols] = color
    return new_image


def paste_over(i1, i2, binary_threshold=0.1):
    # first create a mask for everywhere i1 > binary_threshold, and zero out those pixels in i2, then add.
    mask = np.any(i1 > binary_threshold, axis=2)
    i2[mask] = 0
    return i2 + i1


def adjust_lightness(color, amount=1.0):
    """
    Adjusts the brightness/lightness of a color
    Args:
        color: any valid matplotlib color, could be hex string, or tuple, etc.
        amount: 1 means no change, less than 1 is darker, more than 1 is brighter

    Returns:

    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def adjust_lightness_msg(color: ColorRGBA, amount=1.0):
    """
    Adjusts the brightness/lightness of a color
    Args:
        color:
        amount: 1 means no change, less than 1 is darker, more than 1 is brighter

    Returns:

    """
    mpl_color = [color.r, color.g, color.b, color.a]
    try:
        c = mc.cnames[mpl_color]
    except:
        c = mpl_color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    new_c = colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
    return ColorRGBA(*new_c, color.a)


def get_rotation(xticklabels: Iterable[str]):
    max_len = max(*[len(l) for l in xticklabels])
    rotation = 1.2 * max_len
    return rotation


def get_figsize(n_elements: int):
    """
    Args:
        n_elements: the number of bars in a bar chart, for example

    Returns:
        width and height of the figure/subplot

    """
    q = 4
    aspect = q / (n_elements + (q - 1))
    w, h = figaspect(aspect)
    return w, h