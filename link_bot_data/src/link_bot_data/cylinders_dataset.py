import gzip
import pathlib
import pickle

import hjson
from torch.utils.data import Dataset


def load_mode_filenames(d: pathlib.Path, filenames_filename: pathlib.Path):
    with filenames_filename.open("r") as filenames_file:
        filenames = [l.strip("\n") for l in filenames_file.readlines()]
    return [d / p for p in filenames]


def load_hjson(path: pathlib.Path):
    with path.open("r") as file:
        data = hjson.load(file)
    return data


def load_metadata(metadata_filename):
    if 'hjson' in metadata_filename.name:
        metadata = load_hjson(metadata_filename)
    elif 'pkl' in metadata_filename.name:
        with metadata_filename.open("rb") as f:
            metadata = pickle.load(f)
    else:
        raise NotImplementedError(f"Unsupported metadata filename {metadata_filename.suffix}")
    metadata['filename'] = metadata_filename.stem
    metadata['example_idx'] = int(metadata_filename.stem[8:])
    metadata['full_filename'] = metadata_filename.as_posix()
    return metadata


def load_gzipped_pickle(filename):
    with gzip.open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def get_filenames(dataset_dirs, mode: str):
    all_filenames = []
    for d in dataset_dirs:
        if mode == 'all':
            all_filenames.extend(load_mode_filenames(d, d / f'train.txt'))
            all_filenames.extend(load_mode_filenames(d, d / f'test.txt'))
            all_filenames.extend(load_mode_filenames(d, d / f'val.txt'))
        elif mode == 'notrain':
            all_filenames.extend(load_mode_filenames(d, d / f'test.txt'))
            all_filenames.extend(load_mode_filenames(d, d / f'val.txt'))
        elif mode == 'notest':
            all_filenames.extend(load_mode_filenames(d, d / f'train.txt'))
            all_filenames.extend(load_mode_filenames(d, d / f'val.txt'))
        else:
            filenames_filename = d / f'{mode}.txt'
            all_filenames.extend(load_mode_filenames(d, filenames_filename))
    all_filenames = sorted(all_filenames)
    return all_filenames


def load_single(metadata_filename: pathlib.Path):
    metadata = load_metadata(metadata_filename)

    data_filename = metadata.pop("data")
    full_data_filename = metadata_filename.parent / data_filename
    if str(data_filename).endswith('.gz'):
        example = load_gzipped_pickle(full_data_filename)
    else:
        with full_data_filename.open("rb") as f:
            example = pickle.load(f)
    example.update(metadata)
    example['metadata'] = metadata
    return example


class MyTorchDataset(Dataset):

    def __init__(self, dataset_dir: pathlib.Path, mode: str):
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.metadata_filenames = get_filenames([dataset_dir], mode)

    def __len__(self):
        return len(self.metadata_filenames)

    def __getitem__(self, idx):
        metadata_filename = self.metadata_filenames[idx]
        example = load_single(metadata_filename)
        return example
