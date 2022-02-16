#!/usr/bin/env python
import argparse
import pathlib

import hjson

from arc_utilities.filesystem_utils import rm_tree
from moonshine.filepath_tools import load_hjson


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=pathlib.Path)

    args = parser.parse_args()

    subdirs_to_remove = ['classifier_datasets', 'training_logdir', 'planning_results']
    for s in subdirs_to_remove:
        d = args.results_dir / s
        if d.exists():
            rm_tree(d)

    logfilename = args.results_dir / 'logfile.hjson'
    log = load_hjson(logfilename)
    keys_to_remove = []
    for k in log.keys():
        if 'iteration ' in k:
            keys_to_remove.append(k)

    for k in keys_to_remove:
        log.pop(k)

    with logfilename.open('w') as f:
        hjson.dump(log, f)


if __name__ == '__main__':
    main()
