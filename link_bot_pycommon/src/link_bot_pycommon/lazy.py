from typing import Callable


class Lazy:
    def __init__(self, my_callable: Callable, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.called = None
        self.class_name = my_callable

    def __getattr__(self, attr):
        if self.called is None:
            self.called = self.class_name(*self.args, **self.kwargs)
        return getattr(self.called, attr)
