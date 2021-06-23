import pathlib
from typing import List

from learn_invariance.new_dynamics_dataset import NewDynamicsDatasetLoader
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_data.new_classifier_dataset import NewClassifierDatasetLoader


def get_dynamics_dataset_loader(dataset_dirs: List[pathlib.Path], **kwargs):
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


def get_classifier_dataset_loader(dataset_dirs: List[pathlib.Path], **kwargs):
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


def guess_dataset_loader(dataset_dir: pathlib.Path, **kwargs):
    for p in dataset_dir.parts:
        if 'classifier' in p:
            return get_classifier_dataset_loader([dataset_dir], **kwargs)
        if 'fwd_model_data' in p:
            return get_dynamics_dataset_loader([dataset_dir], **kwargs)
    raise NotImplementedError()


def guess_dataset_format(dataset_dir: pathlib.Path):
    for e in dataset_dir.iterdir():
        if e.is_file():
            if 'tfrecord' in e.as_posix():
                return 'tfrecord'
            elif 'pkl' in e.as_posix():
                return 'pkl'
        elif e.is_dir():
            for sub_e in e.iterdir():
                if sub_e.is_file():
                    if 'tfrecord' in sub_e.as_posix():
                        return 'tfrecord'
                    elif 'pkl' in sub_e.as_posix():
                        return 'pkl'
    raise NotImplementedError()
