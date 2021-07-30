from time import perf_counter


class MyPeriodicTimer:

    def __init__(self, period: int):
        """

        Args:
            period: seconds
        """
        self.period = period
        self.t0 = perf_counter()

    def __bool__(self):
        dt = perf_counter() - self.t0
        ready = dt > self.period
        if ready:
            self.t0 = perf_counter()
        return ready