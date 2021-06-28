#!/usr/bin/env python3
import argparse
import pathlib
import pickle
import shutil

import pandas as pd

from arc_utilities import ros_init
from arc_utilities.filesystem_utils import get_all_subdirs
from link_bot_planning.analysis.analyze_results import load_table_specs, reduce_metrics3, \
    column_names, load_results
from moonshine.filepath_tools import load_hjson


@ros_init.with_ros("analyze_transfer_planning_metrics")
def main():
    # TODO: re-run this and check that the one new metrics gets results generated
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dirs", type=pathlib.Path, nargs='+')
    parser.add_argument("--regenerate", action='store_true')

    args = parser.parse_args()

    outdir = pathlib.Path('results/transfer_planning_metrics')
    outfile = outdir / 'data.pkl'

    if outfile.exists():
        outfile_bak = outfile.parent / (outfile.name + '.bak')
        shutil.copy(outfile, outfile_bak)

    if not args.regenerate and outfile.exists():
        with outfile.open("rb") as f:
            df = pickle.load(f)
    else:
        df = pd.DataFrame([], columns=column_names, dtype=float)

    results_dirs = get_all_subdirs(args.results_dirs)

    df = load_results(df, results_dirs, outfile)

    method_names = df['method_name'].unique()

    analysis_params = load_hjson(pathlib.Path("analysis_params/env_across_methods.json"))
    tables_config_filename = pathlib.Path("analysis_params/tables_configs/planning_evaluation.hjson")
    table_specs = load_table_specs(tables_config_filename, table_format='simple')

    for spec in table_specs:
        data_for_table = reduce_metrics3(spec.reductions, spec.axes_names, df)
        spec.table.make_table(data_for_table, method_names)
        spec.table.save(outdir)

    for spec in table_specs:
        print()
        spec.table.print()
        print()


if __name__ == '__main__':
    main()
