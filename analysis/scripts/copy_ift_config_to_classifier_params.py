#!/usr/bin/env python

import argparse
import pathlib

import hjson

from moonshine.filepath_tools import load_hjson


def main():
    d = pathlib.Path('/media/shared/ift/untrained-full-retrain_1630704441_d5cca1dfa1')
    log = load_hjson(d / 'logfile.hjson')
    uuid = log['ift_uuid']
    # ift_config = log['ift_config']

    for subdir in (d / 'planning_results').iterdir():
        p = load_hjson(subdir / 'metadata.hjson')
        # p['ift_config'] = ift_config
        p['ift_uuid'] = uuid
        with (subdir / 'metadata.hjson').open('w') as f:
            hjson.dump(p, f)


if __name__ == '__main__':
    main()
