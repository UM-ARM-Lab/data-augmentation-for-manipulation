#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arc_utilities import ros_init
from moonshine.filepath_tools import load_hjson
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


@ros_init.with_ros("ift_classifier_metrics")
def main():
    pd.options.display.max_rows = 999

    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=pathlib.Path)

    args = parser.parse_args()

    plt.style.use('slides')

    mistakes_over_time_filename = args.results_dir / 'mistakes_over_time.hjson'
    mistakes_over_time = load_hjson(mistakes_over_time_filename)
    mistakes_over_time = {int(k): v for k, v in mistakes_over_time.items()}

    plt.figure()
    cumsum = [0] * len(mistakes_over_time[0].keys())
    for start_i, mistakes_over_time_i in sorted(mistakes_over_time.items()):
        start_i = int(start_i)
        iterations_i = [int(i) for i in mistakes_over_time_i.keys()]
        mistakes_i = [int(m) for m in mistakes_over_time_i.values()]
        mistakes_i_cumsum = np.array(cumsum[start_i:]) + mistakes_i

        for ii, m in zip(iterations_i, mistakes_i):
            cumsum[ii] += m
        plt.plot(iterations_i, mistakes_i_cumsum, label=f'iter={start_i}')
    plt.xlabel("iteration")
    plt.ylabel("num mistakes")
    plt.title("mistakes over time")
    plt.show()


if __name__ == '__main__':
    main()
