#!/usr/bin/env python
import pathlib
import pickle

from link_bot_pycommon.serialization import load_gzipped_pickle


def main():
    root = pathlib.Path(".")
    gzs = list(root.glob("*.gz"))
    for gz in gzs:
        new_gz = pathlib.Path(gz.name.split(".")[0] + ".data.pkl")
        print(f"{gz.as_posix()}-->{new_gz.as_posix()}")
        data = load_gzipped_pickle(gz)
        with new_gz.open("wb") as f:
            pickle.dump(data, f)
        with pathlib.Path(gz.stem).open("rb") as f:
            metadata = pickle.load(f)
        metadata['data'] = new_gz
        with pathlib.Path(gz.stem).open("wb") as f:
            pickle.dump(metadata, f)


if __name__ == '__main__':
    main()
