import pathlib
from typing import List

from learn_invariance.new_dynamics_dataset import NewDynamicsDatasetLoader
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_data.new_classifier_dataset import NewClassifierDatasetLoader


def load_dynamics_dataset(dataset_dirs: List[pathlib.Path], **kwargs):
    d_for_checking_type = dataset_dirs[0]
    for e in d_for_checking_type.iterdir():
        if e.is_file():
            if 'tfrecord' in e.as_posix():
                return DynamicsDatasetLoader(dataset_dirs, **kwargs)
            elif 'pkl' in e.as_posix():
                return NewDynamicsDatasetLoader(dataset_dirs, **kwargs)
        elif e.is_dir():
            for sub_e in e.iterdir():
                if sub_e.is_file():
                    if 'tfrecord' in sub_e.as_posix():
                        return DynamicsDatasetLoader(dataset_dirs, **kwargs)
                    elif 'pkl' in sub_e.as_posix():
                        return NewDynamicsDatasetLoader(dataset_dirs, **kwargs)
    raise NotImplementedError()


def load_classifier_dataset(dataset_dirs: List[pathlib.Path], **kwargs):
    d_for_checking_type = dataset_dirs[0]
    for e in d_for_checking_type.iterdir():
        if e.is_file():
            if 'tfrecord' in e.as_posix():
                return ClassifierDatasetLoader(dataset_dirs, **kwargs)
            elif 'pkl' in e.as_posix():
                return NewClassifierDatasetLoader(dataset_dirs, **kwargs)
        elif e.is_dir():
            for sub_e in e.iterdir():
                if sub_e.is_file():
                    if 'tfrecord' in sub_e.as_posix():
                        return ClassifierDatasetLoader(dataset_dirs, **kwargs)
                    elif 'pkl' in sub_e.as_posix():
                        return NewClassifierDatasetLoader(dataset_dirs, **kwargs)
    raise NotImplementedError()
