import pathlib
import pickle

from link_bot_data.dataset_utils import merge_hparams_dicts
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_hjson


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


class NewDynamicsDatasetLoader:

    def __init__(self, dataset_dirs, mode: str):
        self.dataset_dirs = dataset_dirs
        self.hparams = merge_hparams_dicts(dataset_dirs)
        self.mode = mode
        self.filenames = get_filenames(self.dataset_dirs, self.mode)
        self.scenario = get_scenario(self.hparams['scenario'])

    def __iter__(self):
        for metadata_filename in self.filenames:
            metadata = load_hjson(metadata_filename)
            data_filename = metadata.pop("data")
            full_data_filename = metadata_filename.parent / data_filename
            with full_data_filename.open("rb") as example_file:
                example = pickle.load(example_file)
            example.update(metadata)
            yield example

    def __len__(self):
        return len(self.filenames)
