import collections
import gzip
import multiprocessing
import pathlib
import pickle
from functools import lru_cache
from typing import Optional, List, Dict, OrderedDict

import numpy as np
import torch
from torch._six import string_classes
from torch.utils.data._utils.collate import default_collate_err_msg_format, np_str_obj_array_pattern

from cylinders_simple_demo.utils.numpify import numpify
from cylinders_simple_demo.utils.utils import load_hjson


def coerce_types(d: Dict):
    """
    Converts the types of things in the dict to whatever we want for saving it
    Args:
        d:

    Returns:

    """
    out_d = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            if v.dtype == np.float64:
                out_d[k] = v.astype(np.float32)
            else:
                out_d[k] = v
        elif isinstance(v, np.float64):
            out_d[k] = np.float32(v)
        elif isinstance(v, np.float32):
            out_d[k] = v
        elif isinstance(v, np.int64):
            out_d[k] = v
        elif isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
            if isinstance(v, np.ndarray):
                if v.dtype == np.float64:
                    out_d[k] = v.astype(np.float32)
                else:
                    out_d[k] = v
            elif isinstance(v, np.float64):
                out_d[k] = np.float32(v)
            else:
                out_d[k] = v
        elif isinstance(v, str):
            out_d[k] = v
        elif isinstance(v, bytes):
            out_d[k] = v
        elif isinstance(v, int):
            out_d[k] = v
        elif isinstance(v, float):
            out_d[k] = v
        elif isinstance(v, list):
            v0 = v[0]
            if isinstance(v0, int):
                out_d[k] = np.array(v)
            elif isinstance(v0, float):
                out_d[k] = np.array(v, dtype=np.float32)
            elif isinstance(v0, str):
                out_d[k] = v
            elif isinstance(v0, list):
                v00 = v0[0]
                if isinstance(v00, int):
                    out_d[k] = np.array(v)
                elif isinstance(v00, float):
                    out_d[k] = np.array(v, dtype=np.float32)
                elif isinstance(v00, str):
                    out_d[k] = v
            elif isinstance(v0, np.ndarray):
                if v0.dtype == np.float64:
                    out_d[k] = np.array(v).astype(np.float32)
                else:
                    out_d[k] = np.array(v)
            else:
                raise NotImplementedError(f"{k} {type(v)} {v}")
        elif isinstance(v, dict):
            out_d[k] = coerce_types(v)
        else:
            raise NotImplementedError(f"{k} {type(v)}")
    assert len(out_d) == len(d)
    return out_d


def pkl_write_example(full_output_directory, example, traj_idx, extra_metadata_keys: Optional[List[str]] = None):
    metadata_filename = index_to_filename('.pkl', traj_idx)
    full_metadata_filename = full_output_directory / metadata_filename
    example_filename = index_to_filename('.pkl.gz', traj_idx)
    full_example_filename = full_output_directory / example_filename

    if 'metadata' in example:
        metadata = example.pop('metadata')
    else:
        metadata = {}
    metadata['data'] = example_filename
    if extra_metadata_keys is not None:
        for k in extra_metadata_keys:
            metadata[k] = example.pop(k)

    metadata = coerce_types(metadata)
    with full_metadata_filename.open("wb") as metadata_file:
        pickle.dump(metadata, metadata_file)

    example = coerce_types(example)
    dump_gzipped_pickle(example, full_example_filename)

    return full_example_filename, full_metadata_filename


def index_to_filename(file_extension, traj_idx):
    new_filename = f"example_{traj_idx:08d}{file_extension}"
    return new_filename


def dump_gzipped_pickle(data, filename):
    while True:
        try:
            with gzip.open(filename, 'wb') as data_file:
                pickle.dump(data, data_file)
            return
        except KeyboardInterrupt:
            pass


def load_gzipped_pickle(filename):
    with gzip.open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def load_metadata(metadata_filename):
    if 'hjson' in metadata_filename.name:
        metadata = load_hjson(metadata_filename)
    elif 'pkl' in metadata_filename.name:
        with metadata_filename.open("rb") as f:
            metadata = pickle.load(f)
    else:
        raise NotImplementedError()
    metadata['filename'] = metadata_filename.stem
    metadata['example_idx'] = int(metadata_filename.stem[8:])
    metadata['full_filename'] = metadata_filename.as_posix()
    return metadata


@lru_cache
def load_single(metadata_filename: pathlib.Path, no_update_with_metadata=False):
    metadata = load_metadata(metadata_filename)

    data_filename = metadata.pop("data")
    full_data_filename = metadata_filename.parent / data_filename
    if str(data_filename).endswith('.gz'):
        example = load_gzipped_pickle(full_data_filename)
    else:
        with full_data_filename.open("rb") as f:
            example = pickle.load(f)
    example['metadata'] = metadata
    if not no_update_with_metadata:
        example.update(metadata)
    return example


def load_mode_filenames(d: pathlib.Path, filenames_filename: pathlib.Path):
    with filenames_filename.open("r") as filenames_file:
        filenames = [l.strip("\n") for l in filenames_file.readlines()]
    return [d / p for p in filenames]


def get_filenames(d, mode: str):
    all_filenames = []
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


def pprint_example(example):
    for k, v in example.items():
        if hasattr(v, 'shape'):
            print(k, v.shape, v.dtype)
        elif isinstance(v, OrderedDict):
            print(k, numpify(v))
        else:
            print(k, type(v))


def remove_keys(*keys):
    def _remove_keys(example):
        for k in keys:
            example.pop(k, None)
        return example

    return _remove_keys


def get_num_workers(batch_size):
    return min(batch_size, multiprocessing.cpu_count())


def my_collate(batch):
    """ Copied from the default_collate """
    if isinstance(batch, collections.abc.Mapping):
        return batch

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).view(-1, *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                # NOTE: the original code through an exception here. Instead we return the original input as numpy
                return np.array(batch)

            return my_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(np.array(batch))
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        common_keys = []
        for key in elem:
            is_common = True
            for b in batch:
                if key not in b:
                    is_common = False
                    break
            if is_common:
                common_keys.append(key)
        return {key: my_collate([d[key] for d in batch]) for key in common_keys}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(my_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [my_collate(samples) for samples in transposed]
    raise TypeError(default_collate_err_msg_format.format(elem_type))
