import collections

import numpy as np
import torch
from torch._six import string_classes
from torch.utils.data import Subset, ConcatDataset
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format

import rospy


def dataset_repeat(dataset, repeat: int):
    if repeat is None:
        return dataset

    dataset_repeated = ConcatDataset([dataset for _ in range(repeat)])
    return dataset_repeated


def dataset_take(dataset, take):
    if take is None:
        return dataset

    dataset_take = Subset(dataset, range(min(take, len(dataset))))
    return dataset_take


def dataset_skip(dataset, skip):
    if skip is None:
        return dataset

    dataset_take = Subset(dataset, range(skip, len(dataset)))
    return dataset_take


def dataset_shard(dataset, shard):
    if shard is None:
        return dataset

    dataset_take = Subset(dataset, range(0, len(dataset), shard))
    return dataset_take


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
            out = elem.new(storage)
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
    elif isinstance(elem, rospy.Message):
        return batch
    raise TypeError(default_collate_err_msg_format.format(elem_type))
