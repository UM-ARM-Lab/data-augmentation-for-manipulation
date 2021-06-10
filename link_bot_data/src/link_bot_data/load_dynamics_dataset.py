import pathlib
from typing import List

from learn_invariance.new_dynamics_dataset import NewDynamicsDatasetLoader
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader


def load_dynamics_dataset(dataset_dir: List[pathlib.Path]):
    d_for_checking_type = dataset_dir[0]
    for e in d_for_checking_type.iterdir():
        if e.is_file():
            if 'tfrecord' in e.as_posix():
                return DynamicsDatasetLoader(dataset_dir)
            elif 'pkl' in e.as_posix():
                return NewDynamicsDatasetLoader(dataset_dir)
        elif e.is_dir():
            for sub_e in e.iterdir():
                if sub_e.is_file():
                    if 'tfrecord' in sub_e.as_posix():
                        return DynamicsDatasetLoader(dataset_dir)
                    elif 'pkl' in sub_e.as_posix():
                        return NewDynamicsDatasetLoader(dataset_dir)
    raise NotImplementedError()
