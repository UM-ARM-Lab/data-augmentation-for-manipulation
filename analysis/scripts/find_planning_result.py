#!/usr/bin/env python

import argparse
import pathlib

import link_bot_pycommon.pycommon
from analysis.analyze_results import load_planning_results
from analysis.results_utils import get_all_results_subdirs


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    root = pathlib.Path("/media/shared/planning_results")
    results_dirs = get_all_results_subdirs(root)

    def query_fun(row):
        match = all([
            row['target_env'] == 'swap_straps_no_recovery3',
            row['classifier_source_env'] == 'floating_boxes',
            row['recovery_name'] == 'recovery_trials/random',
        ])
        return match

    for d in results_dirs:
        try:
            df = load_planning_results([d], regenerate=False, progressbar=False)
        except Exception:
            df = load_planning_results([d], regenerate=True)
        if query_fun(df.loc[0]):
            print(link_bot_pycommon.pycommon.as_posix())


if __name__ == '__main__':
    main()
