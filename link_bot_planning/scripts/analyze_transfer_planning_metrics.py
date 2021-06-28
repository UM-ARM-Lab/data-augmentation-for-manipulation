#!/usr/bin/env python3
import argparse
import pathlib
import pickle
import shutil
from typing import List

import pandas as pd
from progressbar import progressbar

from arc_utilities import ros_init
from arc_utilities.filesystem_utils import get_all_subdirs
from link_bot_data.progressbar_widgets import mywidgets
from link_bot_planning.analysis.analyze_results import load_table_specs, metrics_funcs, metrics_names
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.pandas_utils import df_append
from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.filepath_tools import load_hjson

column_names = [
    'method_name',
    'seed',
    'ift_iteration',
    'trial_idx',
    'uuid',
]
column_names += metrics_names


def load_results(df, results_dirs: List[pathlib.Path], outfile):
    outfile_tmp = outfile.parent / (outfile.name + '.tmp')

    # TODO: refactor into a generator that pairs metadata with datum directly
    # then we can put a progressbar on the whole thing very neatly
    for d in results_dirs:
        metrics_filenames = list(d.glob("*_metrics.pkl.gz"))
        metadata_filename = d / 'metadata.hjson'
        metadata = load_hjson(metadata_filename)
        for file_idx, metrics_filename in enumerate(progressbar(metrics_filenames, widgets=mywidgets)):
            # load and compute metrics
            datum = load_gzipped_pickle(metrics_filename)

            already_exists = datum['uuid'] in df['uuid'].unique()
            if already_exists:
                continue

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

            df = df_append(df, row)

        with outfile_tmp.open("wb") as f:
            pickle.dump(df, f)
    # if everything went well now overwrite the input file
    with outfile.open("wb") as f:
        pickle.dump(df, f)
    return df


def reduce_metrics(reductions: List[List], axis_names: List[str], metrics: pd.DataFrame):
    reduced_metrics = []
    for reduction in reductions:
        metric_i = metrics.copy()
        for reduction_step in reduction:
            if 'group_by' in reduction_step:
                data_for_axis_groupby = metric_i.groupby(reduction_step['group_by'])
                metric_i = data_for_axis_groupby.agg({reduction_step['metric']: reduction_step['agg']})
            elif 'keys' in reduction_step:
                metric_i = metric_i[reduction_step['keys'] + [reduction_step['metric']]]
            else:
                raise NotImplementedError(reduction_step)
        reduced_metrics.append(metric_i)

    reduced_metrics = pd.concat(reduced_metrics, axis=1)
    # columns = dict(zip([r[0] for r in reductions], axis_names))
    # reduced_metrics.rename(columns=columns, inplace=True)

    return reduced_metrics


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
        df = pd.DataFrame([], columns=column_names)

    results_dirs = get_all_subdirs(args.results_dirs)

    # df = load_results(df, results_dirs, outfile)
    method_names = df['method_name'].unique()

    analysis_params = load_hjson(pathlib.Path("analysis_params/env_across_methods.json"))
    tables_config_filename = pathlib.Path("analysis_params/tables_configs/planning_evaluation.hjson")
    table_specs = load_table_specs(tables_config_filename, table_format='simple')

    for spec in table_specs:
        data_for_table = reduce_metrics(spec.reductions, spec.axes_names, df)
        spec.table.make_table(data_for_table, method_names)
        spec.table.save(outdir)

    for spec in table_specs:
        print()
        spec.table.print()
        print()


if __name__ == '__main__':
    main()
