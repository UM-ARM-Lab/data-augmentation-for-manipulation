#!/usr/bin/env python
import pickle
import os
import subprocess
import argparse
import pathlib
import tempfile

import hjson


def pkledit(pkl: pathlib.Path):
    with pkl.open("rb") as f:
        d = pickle.load(f)

    tmpfile = tempfile.NamedTemporaryFile()
    with open(tmpfile.name, 'w') as f:
        hjson.dump(d, f)

    editor = os.environ.get('EDITOR', 'vim')
    subprocess.run([editor, tmpfile.name])

    with open(tmpfile.name, 'r') as f:
        d = hjson.load(f)

    with pkl.open("wb") as f:
        pickle.dump(d, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pkls', type=pathlib.Path, nargs='+')

    args = parser.parse_args()

    for pkl in args.pkls:
        pkledit(pkl)


if __name__ == '__main__':
    main()
