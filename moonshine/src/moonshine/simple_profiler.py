from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt


class SimpleProfiler:

    def __init__(self):
        self.fig = plt.figure()
        self.ax = plt.gca()
        self.ax.set_xlabel("time (ms)")
        self.ax.set_ylabel("frequency")
        self.t0 = perf_counter()
        self.dts = []

    def start(self):
        self.t0 = perf_counter()

    def stop(self):
        now = perf_counter()
        dt = now - self.t0
        self.dts.append(dt)

    def lap(self):
        now = perf_counter()
        dt = now - self.t0
        self.dts.append(dt)
        self.t0 = now

    def get_results_str(self):
        def ninety_fifth_percentile(x):
            return np.percentile(x, 90)

        def fifth_percentile(x):
            return np.percentile(x, 5)

        measurements = [
            np.mean,
            np.median,
            np.min,
            np.max,
            fifth_percentile,
        ]
        s = ""
        for measurement in measurements:
            s += f"{measurement.__name__}: {measurement(self.dts) * 1e3:.4f}ms "
        return s

    def profile(self, max_iters, f, *args, **kwargs):
        # the first call often is slow for caching reasons, but we don't want to measure that
        f(*args, **kwargs)
        f(*args, **kwargs)
        f(*args, **kwargs)
        f(*args, **kwargs)

        self.start()

        overall_t0 = perf_counter()
        for i in range(max_iters):
            now = perf_counter()

            f(*args, **kwargs)
            overall_dt = now - overall_t0
            if overall_dt > 30:
                break

            self.lap()

        self.ax.hist(self.dts, label=f.__name__, alpha=0.3, bins=20)

        return self.get_results_str()

    def __str__(self):
        return self.get_results_str()
