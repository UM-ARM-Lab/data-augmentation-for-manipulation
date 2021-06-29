#!/usr/bin/env python
import argparse
import pathlib
import pickle
import shutil

import pandas as pd

from arc_utilities import ros_init
from link_bot_planning.analysis.results_utils import get_all_results_subdirs
from link_bot_planning.analysis.analyze_results import load_results, column_names
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


@ros_init.with_ros("analyse_planning_results")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='results directory', type=pathlib.Path,
                        default=pathlib.Path("/media/shared/planning_results"))
    parser.add_argument('--regenerate', action='store_true')

    args = parser.parse_args()

    outfile = args.root / 'data.pkl'

    if outfile.exists():
        outfile_bak = outfile.parent / (outfile.name + '.bak')
        shutil.copy(outfile, outfile_bak)

    if not args.regenerate and outfile.exists():
        with outfile.open("rb") as f:
            df = pickle.load(f)
    else:
        df = pd.DataFrame([], columns=column_names, dtype=float)

    results_dirs = get_all_results_subdirs(args.root)
    print("Found:")
    for d in results_dirs:
        print(d.as_posix())
    load_results(df, results_dirs, outfile)


if __name__ == '__main__':
    main()
