import pathlib
import shutil
from typing import List, Optional

import numpy as np

from link_bot_data.dataset_utils import DEFAULT_VAL_SPLIT, DEFAULT_TEST_SPLIT


class FilesDataset:

    def __init__(self, root_dir: pathlib.Path, val_split: Optional[float] = DEFAULT_VAL_SPLIT,
                 test_split: Optional[float] = DEFAULT_TEST_SPLIT):
        self.root_dir = root_dir
        self.paths = []
        self.test_split = test_split if test_split is not None else DEFAULT_TEST_SPLIT
        self.val_split = val_split if val_split is not None else DEFAULT_VAL_SPLIT

    def add(self, full_filename: pathlib.Path):
        self.paths.append(full_filename)

    def split(self, shuffle=False):
        rng = np.random.RandomState(0)

        make_subdir(self.root_dir, 'train')
        make_subdir(self.root_dir, 'val')
        make_subdir(self.root_dir, 'test')

        n_files = len(self.paths)
        n_validation = int(self.val_split * n_files)
        n_testing = int(self.test_split * n_files)

        if shuffle:
            rng.shuffle(self.paths)

        val_files = self.paths[0:n_validation]
        self.paths = self.paths[n_validation:]

        if shuffle:
            rng.shuffle(self.paths)

        test_files = self.paths[0:n_testing]
        train_files = self.paths[n_testing:]

        move_files(self.root_dir, train_files, 'train')
        move_files(self.root_dir, test_files, 'test')
        move_files(self.root_dir, val_files, 'val')


def make_subdir(root_dir: pathlib.Path, subdir: str):
    subdir_path = root_dir / subdir
    subdir_path.mkdir(exist_ok=True)


def move_files(root_dir: pathlib.Path, files: List[pathlib.Path], mode: str):
    for file in files:
        out = root_dir / mode / file.name
        shutil.move(file, out)
