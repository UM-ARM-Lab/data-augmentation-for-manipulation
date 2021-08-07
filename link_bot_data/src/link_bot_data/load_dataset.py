import pathlib
from typing import List

from learn_invariance.new_dynamics_dataset import NewDynamicsDatasetLoader
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_data.dataset_utils import compute_batch_size_for_n_examples
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_data.new_classifier_dataset import NewClassifierDatasetLoader


def get_dynamics_dataset_loader(dataset_dirs: List[pathlib.Path], **kwargs):
    if isinstance(dataset_dirs, pathlib.Path):
        dataset_dirs = [dataset_dirs]
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
    if isinstance(dataset_dirs, pathlib.Path):
        dataset_dirs = [dataset_dirs]
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
    return 'pkl'


def guess_dataset_size(dataset_dir: pathlib.Path):
    format = guess_dataset_format(dataset_dir)
    if format == 'tfrecord':
        d = dataset_dir / 'val'
        if d.is_dir():
            n_examples = len(list(d.glob("*.tfrecords")))
            return n_examples
    elif format == 'pkl':
        n_examples = len(list(dataset_dir.glob("*.pkl")))
        return n_examples
    else:
        raise NotImplementedError(format)


def compute_batch_size(dataset_dirs: List[pathlib.Path], max_batch_size: int):
    total_examples = 0
    for dataset_dir in dataset_dirs:
        # assumes validation is smaller than or the same size as train
        total_examples += guess_dataset_size(dataset_dir)
    return compute_batch_size_for_n_examples(total_examples, max_batch_size)