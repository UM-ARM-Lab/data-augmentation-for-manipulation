import gzip
import pickle
from typing import Optional, List, Dict

import numpy as np
import tensorflow as tf
import torch


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
        elif isinstance(v, tf.Tensor):
            v = v.numpy()
            if isinstance(v, np.ndarray):
                if v.dtype == np.float64:
                    out_d[k] = v.astype(np.float32)
                else:
                    out_d[k] = v
            elif isinstance(v, np.float64):
                out_d[k] = np.float32(v)
            else:
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
            elif isinstance(v0, tf.Tensor):
                out_d[k] = tf.convert_to_tensor(v)
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


def index_to_record_name(traj_idx):
    return index_to_filename('.tfrecords', traj_idx)


def index_to_filename2(traj_idx, save_format):
    if save_format == 'pkl':
        return index_to_filename('.pkl', traj_idx)
    elif save_format == 'tfrecord':
        return index_to_record_name(traj_idx)


def count_up_to_next_record_idx(full_output_directory):
    record_idx = 0
    while True:
        record_filename = index_to_record_name(record_idx)
        full_filename = full_output_directory / record_filename
        if not full_filename.exists():
            break
        record_idx += 1
    return record_idx


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
