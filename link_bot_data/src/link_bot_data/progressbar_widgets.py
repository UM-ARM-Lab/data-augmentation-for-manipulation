import progressbar
from progressbar.widgets import FormatWidgetMixin, WidgetBase


class Total(FormatWidgetMixin, WidgetBase):
    """Displays the total"""

    def __init__(self, format='%(max_value)d', **kwargs):
        FormatWidgetMixin.__init__(self, format=format, **kwargs)
        WidgetBase.__init__(self, format=format, **kwargs)

    def __call__(self, progress, data, format=None):
        return FormatWidgetMixin.__call__(self, progress, data, format)


mywidgets = [
    progressbar.Percentage(), ' ',
    progressbar.Counter(), '/', Total(), ' ',
    progressbar.Bar(),
    ' (', progressbar.AdaptiveETA(), ') ',
]
