import logging
import pathlib
import pickle
from copy import deepcopy
from typing import List, Dict, Optional

import pandas as pd
import tabulate
from tqdm import tqdm

from analysis.figspec import DEFAULT_AXES_NAMES, FigSpec, TableSpec
# noinspection PyUnresolvedReferences
from analysis.results_figures import *
from analysis.results_metrics import metrics_funcs, load_analysis_hjson
from analysis.results_metrics import metrics_names
# noinspection PyUnresolvedReferences
from analysis.results_tables import *
from analysis.results_utils import get_all_results_subdirs
from link_bot_pycommon.get_scenario import get_scenario_cached
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.filepath_tools import load_hjson

logger = logging.getLogger(__file__)

column_names = [
                   'data_filename',
                   'results_folder_name',
                   'method_name',
                   'seed',
                   'ift_iteration',
                   'trial_idx',
                   'uuid',
               ] + metrics_names


def load_fig_specs(analysis_params, figures_config: pathlib.Path):
    figures_config = load_hjson(figures_config)
    figspecs = []
    for fig_config in figures_config:
        figure_type = eval(fig_config.pop('type'))
        reductions = fig_config.pop('reductions')
        axes_names = DEFAULT_AXES_NAMES

        fig = figure_type(analysis_params, **fig_config)

        figspec = FigSpec(fig=fig, reductions=reductions, axes_names=axes_names)
        figspecs.append(figspec)
    return figspecs


def load_table_specs(tables_config: pathlib.Path, table_format: str):
    tables_conf = load_analysis_hjson(tables_config)
    return make_table_specs(table_format, tables_conf)


def make_table_specs(table_format: str, tables_conf: List[Dict]):
    tablespecs = []
    tables_conf = deepcopy(tables_conf)
    for table_conf in tables_conf:
        table_type = eval(table_conf.pop('type'))
        reductions = table_conf.pop('reductions')
        axes_names = DEFAULT_AXES_NAMES

        table = table_type(table_format=table_format, **table_conf)

        tablespec = TableSpec(table=table, reductions=reductions, axes_names=axes_names)
        tablespecs.append(tablespec)
    return tablespecs


def reduce_planning_metrics(reductions: List[List], metrics: pd.DataFrame):
    reduced_metrics = []
    for reduction in reductions:
        metric_i = metrics.copy()
        for reduction_step in reduction:
            group_by, metric, agg = reduction_step
            assert metric is not None
            if group_by is None or len(group_by) == 0:
                metric_i = metric_i.agg({metric: agg})
            elif group_by is not None and agg is not None:
                if metric in metric_i.index.names:
                    assert len(metric_i.columns) == 1
                    metric_i.columns = ['tmp']
                    metric_i = metric_i.groupby(group_by, dropna=False).agg({'tmp': agg})
                    metric_i.columns = [metric]
                else:
                    metric_i = metric_i.groupby(group_by, dropna=False).agg({metric: agg})
            elif group_by is not None and agg is None:
                metric_i.set_index(group_by, inplace=True)
                metric_i = metric_i[metric]
            else:
                raise NotImplementedError()
        reduced_metrics.append(metric_i)

    reduced_metrics = pd.concat(reduced_metrics, axis=1)
    return reduced_metrics


def load_planning_results(results_dirs: List[pathlib.Path], regenerate: bool = False, progressbar: bool = True):
    dfs = []
    results_dirs_gen = tqdm(results_dirs, desc='results dirs') if progressbar else results_dirs
    for d in results_dirs_gen:
        data_filenames = list(d.glob("*_metrics.pkl.gz"))
        df_filename = d / 'df.pkl'
        metadata_filename = d / 'metadata.hjson'
        metadata = load_hjson(metadata_filename)
        if not df_filename.exists() or regenerate:
            scenario = get_scenario_cached(metadata['planner_params']['scenario'])
            data = []
            data_gen = tqdm(data_filenames, desc='results files') if progressbar else data_filenames
            for data_filename in data_gen:
                datum = load_gzipped_pickle(data_filename)
                row = make_row(datum, data_filename, metadata, scenario)

                data.append(row)
            df_i = pd.DataFrame(data)
            with df_filename.open("wb") as f:
                pickle.dump(df_i, f)
        else:
            with df_filename.open("rb") as f:
                df_i = pickle.load(f)
        dfs.append(df_i)

    df = pd.concat(dfs)
    df.columns = column_names
    return df


def make_row(datum: Dict, data_filename: pathlib.Path, metadata: Dict, scenario: ScenarioWithVisualization):
    metrics_values = [metric_func(data_filename, scenario, metadata, datum) for metric_func in metrics_funcs]
    trial_idx = datum['trial_idx']
    try:
        seed_guess = datum['steps'][0]['planning_query'].seed - 100000 * trial_idx
    except (KeyError, IndexError):
        seed_guess = 0
    seed = datum.get('seed', seed_guess)

    results_folder_name = guess_results_folder_name(data_filename)

    row = [
        data_filename.as_posix(),
        results_folder_name,
        metadata['planner_params']['method_name'],
        seed,
        metadata.get('ift_iteration', 0),
        trial_idx,
        datum['uuid'],
    ]
    row += metrics_values
    return row


def guess_results_folder_name(data_filename):
    results_folders = data_filename.parts[:-1]
    results_folder_name = pathlib.Path(*results_folders[-2:]).as_posix()
    return results_folder_name


def generate_tables(df: pd.DataFrame, outdir: Optional[pathlib.Path], table_specs):
    for spec in table_specs:
        data_for_table = reduce_planning_metrics(spec.reductions, df)
        spec.table.make_table(data_for_table)
        if outdir is not None:
            spec.table.save(outdir)
    for spec in table_specs:
        print()
        spec.table.print()
        print()
    if outdir is not None:
        tables_outfilename = outdir / 'tables.txt'
        with tables_outfilename.open("w") as tables_outfile:
            for spec in table_specs:
                tables_outfile.write(spec.table.table)
                tables_outfile.write('\n\n\n')


def planning_results(results_dirs, regenerate=False, latex=False, tables_config=None):
    # The default for where we write results
    outdir = results_dirs[0]

    print(f"Writing analysis to {outdir}")

    if latex:
        table_format = 'latex_raw'
    else:
        table_format = tabulate.simple_separated_format("\t")

    results_dirs = get_all_results_subdirs(results_dirs)
    df = load_planning_results(results_dirs, regenerate=regenerate)
    df.to_csv("/media/shared/analysis/tmp_results.csv")

    if tables_config is not None:
        table_specs = load_table_specs(tables_config, table_format)
    else:
        table_specs = None

    return outdir, df, table_specs
