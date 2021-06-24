#!/usr/bin/env python3
import argparse
import pathlib
import pickle
from typing import List

import pandas as pd
from progressbar import progressbar

from arc_utilities import ros_init
from arc_utilities.filesystem_utils import get_all_subdirs
from link_bot_data.progressbar_widgets import mywidgets
from link_bot_planning.analysis.analyze_results import load_table_specs, metrics_funcs, metrics_names
from link_bot_planning.analysis.figspec import get_data_for_table
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.filepath_tools import load_hjson


def load_results(data, results_dirs: List[pathlib.Path]):
    column_names = [
        'method_name',
        'seed',
        'ift_iteration',
        'trial_idx',
        'uuid',
    ]
    # TODO: refactor into a generator that pairs metadata with datum directly
    # then we can put a progressbar on the whole thing very neatly
    column_names += metrics_names
    for d in results_dirs:
        metrics_filenames = list(d.glob("*_metrics.pkl.gz"))
        metadata_filename = d / 'metadata.hjson'
        metadata = load_hjson(metadata_filename)
        for file_idx, metrics_filename in enumerate(progressbar(metrics_filenames, widgets=mywidgets)):
            # load and compute metrics
            datum = load_gzipped_pickle(metrics_filename)
            scenario = get_scenario(metadata['planner_params']['scenario'])
            metrics_values = [metric_func(scenario, metadata, datum) for metric_func in metrics_funcs]
            # create and add a row
            row = [
                metadata['planner_params']['method_name'],
                datum.get('seed', 0),
                metadata.get('ift_iteration', 0),
                datum['trial_idx'],
                datum['uuid'],
            ]
            row += metrics_values
            data.append(row)
    return pd.DataFrame(data, columns=column_names)


@ros_init.with_ros("analyze_transfer_planning_metrics")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dirs", type=pathlib.Path, nargs='+')
    parser.add_argument("--regenerate", action='store_true')

    args = parser.parse_args()

    outdir = pathlib.Path('results/transfer_planning_metrics')
    outfile = outdir / 'data.pkl'
    if not args.regenerate and outfile.exists():
        with outfile.open("rb") as f:
            df = pickle.load(f)
    else:
        df = []

    results_dirs = get_all_subdirs(args.results_dirs)
    df = load_results(df, results_dirs)

    with outfile.open("wb") as f:
        pickle.dump(df, f)

    method_names = df['method_names']

    analysis_params = load_hjson(pathlib.Path("analysis_params/env_across_methods.json"))
    tables_config = load_hjson(pathlib.Path("analysis_params/tables_configs/planning_evaluation.hjson"))
    table_specs = load_table_specs(analysis_params, tables_config, table_format='simple')

    for spec in table_specs:
        data_for_table = get_data_for_table(spec, df)
        spec.table.make_table(data_for_table, method_names)
        spec.table.save(outdir)

    for spec in table_specs:
        print()
        spec.table.print()
        print()


if __name__ == '__main__':
    main()
