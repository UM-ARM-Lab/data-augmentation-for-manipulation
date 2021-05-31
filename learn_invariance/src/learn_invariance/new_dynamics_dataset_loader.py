import pathlib
import random

from link_bot_data.dataset_utils import merge_hparams_dicts, batch_sequence
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.filepath_tools import load_hjson
from moonshine.moonshine_utils import batch_examples_dicts


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

    def __init__(self, dataset_dirs, mode: str, batch_size: int, shuffle: bool = False):
        self.dataset_dirs = dataset_dirs
        self.hparams = merge_hparams_dicts(dataset_dirs)
        self.mode = mode
        self.shuffle = shuffle
        self.filenames = get_filenames(self.dataset_dirs, self.mode)
        if self.shuffle:
            random.shuffle(self.filenames)
        self.scenario = get_scenario(self.hparams['scenario'])
        self.batch_size = batch_size
        self.filenames_batched = list(batch_sequence(self.filenames, batch_size))
        assert len(self.filenames_batched) > 0

    def __iter__(self):
        for metadata_filename_batch in self.filenames_batched:
            examples = []
            for metadata_filename in metadata_filename_batch:
                metadata = load_hjson(metadata_filename)
                data_filename = metadata.pop("data")
                full_data_filename = metadata_filename.parent / data_filename
                example = load_gzipped_pickle(full_data_filename)
                example.update(metadata)
                examples.append(example)

            example_batch = batch_examples_dicts(examples)
            yield example_batch

    def __len__(self):
        return len(self.filenames_batched)