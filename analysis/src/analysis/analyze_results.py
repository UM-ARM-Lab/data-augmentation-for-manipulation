import pathlib
import pickle
from typing import List, Dict

import pandas as pd
import tabulate
from tqdm import tqdm

from analysis.results_metrics import metrics_funcs
from analysis.results_metrics import metrics_names
from analysis.results_utils import get_all_results_subdirs
from link_bot_pycommon.get_scenario import get_scenario_cached
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.filepath_tools import load_hjson

column_names = [
                   'data_filename',
                   'results_folder_name',
                   'method_name',
                   'seed',
                   'ift_iteration',
                   'trial_idx',
                   'uuid',
               ] + metrics_names


def load_planning_results(results_dirs: List[pathlib.Path], regenerate: bool = False, progressbar: bool = True):
    dfs = []
    results_dirs_gen = tqdm(results_dirs, desc='results dirs') if progressbar else results_dirs
    for d in results_dirs_gen:
        data_filenames = list(d.glob("*_metrics.pkl.gz"))
        df_filename = d / 'df.pkl'
        metadata_filename = d / 'metadata.hjson'
        metadata = load_hjson(metadata_filename)
        if not df_filename.exists() or regenerate:
            scenario = get_scenario_cached(metadata['planner_params']['scenario'],
                                           params=dict(metadata['planner_params']['scenario_params']))
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

    df = pd.concat(dfs, ignore_index=True)
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
