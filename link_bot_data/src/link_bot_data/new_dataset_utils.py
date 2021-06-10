import pathlib
from typing import Union, List

from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.filepath_tools import load_hjson
from moonshine.moonshine_utils import batch_examples_dicts



UNUSED_COMPAT = None

def load_mode_filenames(d: pathlib.Path, filenames_filename: pathlib.Path):
    with filenames_filename.open("r") as filenames_file:
        filenames = [l.strip("\n") for l in filenames_file.readlines()]
    return [d / p for p in filenames]


def get_filenames(dataset_dirs, mode: str):
    all_filenames = []
    for d in dataset_dirs:
        if mode == 'all':
            all_filenames.extend(load_mode_filenames(d, d / f'train.txt'))
            all_filenames.extend(load_mode_filenames(d, d / f'test.txt'))
            all_filenames.extend(load_mode_filenames(d, d / f'val.txt'))
        else:
            filenames_filename = d / f'{mode}.txt'
            all_filenames.extend(load_mode_filenames(d, filenames_filename))
    all_filenames = sorted(all_filenames)
    return all_filenames


def load_single(metadata_filename: pathlib.Path):
    metadata = load_hjson(metadata_filename)
    data_filename = metadata.pop("data")
    full_data_filename = metadata_filename.parent / data_filename
    example = load_gzipped_pickle(full_data_filename)
    example.update(metadata)
    return example


def load(filenames: Union[pathlib.Path, List[pathlib.Path]]):
    if isinstance(filenames, list):
        examples_i = [load_single(metadata_filename_i) for metadata_filename_i in filenames]
        example = batch_examples_dicts(examples_i)
    else:
        metadata = load_hjson(filenames)
        data_filename = metadata.pop("data")
        full_data_filename = filenames.parent / data_filename
        example = load_gzipped_pickle(full_data_filename)
        example.update(metadata)
    return example