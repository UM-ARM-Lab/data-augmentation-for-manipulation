from matplotlib import colors

from std_msgs.msg import ColorRGBA


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
        color_kwarg = kwargs["color"]
        if isinstance(color_kwarg, ColorRGBA):
            return color_kwarg
        else:
            return ColorRGBA(*colors.to_rgba(kwargs["color"]))
    else:
        r = float(kwargs.get("r", r))
        g = float(kwargs.get("g", g))
        b = float(kwargs.get("b", b))
        a = float(kwargs.get("a", a))
        return ColorRGBA(r, g, b, a)
