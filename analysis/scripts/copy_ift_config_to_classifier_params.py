#!/usr/bin/env python

import argparse
import pathlib

import hjson

from moonshine.filepath_tools import load_hjson


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('d', type=pathlib.Path)
    args = parser.parse_args()
    log = load_hjson(args.d / 'logfile.hjson')
    uuid = log['ift_uuid']
    ift_config = log['ift_config']

    for subdir in (args.d / 'planning_results').iterdir():
        p = load_hjson(subdir / 'metadata.hjson')
        if 'ift_config' in p and 'ift_uuid' in p:
            continue
        p['ift_config'] = ift_config
        p['ift_uuid'] = uuid
        with (subdir / 'metadata.hjson').open('w') as f:
            hjson.dump(p, f)

    for subdir in (args.d / 'training_logdir').iterdir():
        subsubdir = list(subdir.iterdir())[0]
        p = load_hjson(subsubdir / 'params.hjson')
        if 'ift_config' in p and 'ift_uuid' in p:
            continue
        p['ift_config'] = ift_config
        p['ift_uuid'] = uuid
        with (subsubdir / 'params.hjson').open('w') as f:
            hjson.dump(p, f)


if __name__ == '__main__':
    main()
