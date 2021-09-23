import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

import link_bot_pycommon.pycommon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=pathlib.Path, nargs='+')
    args = parser.parse_args()

    _, ax1 = plt.subplots()
    _, ax2 = plt.subplots()
    # _, ax3 = plt.subplots()

    names = []
    all_stdevs = []

    violin_handles = []
    for infile in args.infile:
        name = link_bot_pycommon.pycommon.as_posix().replace("/", '-')
        stdevs = np.load(link_bot_pycommon.pycommon.as_posix())
        names.append(name)
        all_stdevs.append(stdevs)

        print(f'{name} {np.mean(stdevs)} {np.std(stdevs)}')

        ax1.hist(stdevs, bins=100, label=name, alpha=0.4)
        ax1.set_xlabel("ensemble stdev")
        ax1.set_ylabel("count")

        v = ax2.violinplot(stdevs)
        violin_handles.append(v["bodies"][0])
        ax2.set_xlabel("density")
        ax2.set_ylabel("classifier uncertainty")

    ax1.legend()
    ax2.legend(violin_handles, names)
    # ax3.legend()

    plt.show()


if __name__ == '__main__':
    main()
