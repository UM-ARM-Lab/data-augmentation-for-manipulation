#!/usr/bin/env python
import argparse
import pathlib

import hjson

from arc_utilities.path_utils import rm_tree
from moonshine.filepath_tools import load_hjson


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=pathlib.Path)

    args = parser.parse_args()

    rm_tree(args.results_dir / 'classifier_datasets')
    rm_tree(args.results_dir / 'planning_results')
    rm_tree(args.results_dir / 'training_logdir')

    logfilename = args.results_dir / 'logfile.hjson'
    log = load_hjson(logfilename)
    for k in log.keys():
        if 'iteration ' in k:
            log.pop(k)

    with logfilename.open('w') as f:
        hjson.dump(log, f)


if __name__ == '__main__':
    main()
