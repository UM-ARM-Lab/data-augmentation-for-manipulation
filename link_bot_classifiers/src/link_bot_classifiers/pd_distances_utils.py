import pathlib

import numpy as np
from lazyarray import larray

from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.filepath_tools import load_hjson

weights = np.array([
    1.0,
    1.0,
    0.1,
    0.1,
    10.0,
])
too_far = np.array([
    1.0,
    1.0,
    20.0,
    20.0,
    0.05,
])
joints_weights = np.array([
    10.0,
    10.0,
    2.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.5,
    0.1,
    0.1,
    0.1,
    2.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.5,
    0.1,
    0.1,
    0.1,
])


def _stem(p):
    return p.name.split('.')[0]


def load_examples(results_dir, k):
    def _load_examples(i, j):
        if isinstance(i, np.ndarray):
            results = []
            for _i in i:
                _name = f'{_i}-{j}.pkl.gz'
                _results_filename = results_dir / _name
                if _results_filename.exists():
                    _result = load_gzipped_pickle(_results_filename)[k]
                else:
                    _result = None
                results.append(_result)
            return np.array(results)
        elif isinstance(j, np.ndarray):
            results = []
            for _j in j:
                _name = f'{i}-{_j}.pkl.gz'
                _results_filename = results_dir / _name
                if _results_filename.exists():
                    _result = load_gzipped_pickle(_results_filename)[k]
                else:
                    _result = None
                results.append(_result)
            return np.array(results)
        else:
            name = f'{i}-{j}.pkl.gz'
            results_filename = results_dir / name
            if results_filename.exists():
                _result = load_gzipped_pickle(results_filename)[k]
            else:
                _result = None
            return _result

    return _load_examples


def format_distances(results_dir: pathlib.Path, space_idx: int):
    logfilename = results_dir / 'logfile.hjson'
    log = load_hjson(logfilename)
    log.pop("augfiles")
    log.pop("datafiles")
    log.pop("weights")
    n_aug = max([int(k.split('-')[0]) for k in log.keys()]) + 1
    n_data = max([int(k.split('-')[1]) for k in log.keys()]) + 1
    shape = [n_aug, n_data]

    distances_matrix = np.ones(shape) * too_far[space_idx]
    aug_examples_matrix = larray(load_examples(results_dir, 'aug_example'), shape=shape)
    data_examples_matrix = larray(load_examples(results_dir, 'data_example'), shape=shape)

    for k, d in log.items():
        aug_i, data_j = k.split('-')
        aug_i = int(aug_i)
        data_j = int(data_j)
        if d != 'too_far':
            d_space = d[space_idx]
        else:
            d_space = too_far[space_idx]
        distances_matrix[aug_i][data_j] = d_space

    return aug_examples_matrix, data_examples_matrix, distances_matrix


def get_first(m):
    for i, m_i in enumerate(m):
        if m_i is not None:
            return i, m_i
    return -1, None


def space_to_idx(space):
    if space == 'rope':
        space_idx = 0
    elif space == 'robot':
        space_idx = 3
    elif space == 'env':
        space_idx = 4
    else:
        raise NotImplementedError(space)
    return space_idx


def distance_to_score(d):
    return d
    # if d < 1e-5:
    #     return 1e5
    # else:
    #     return 1 / d


def compute_diversity(distances_matrix, aug_examples_matrix, data_examples_matrix):
    diversities = []
    for j in range(distances_matrix.shape[1]):
        distances_for_data_j = distances_matrix[:, j]

        sorted_i = np.argsort(distances_for_data_j)
        best_idx = sorted_i[0]
        for i in sorted_i:
            data_e = data_examples_matrix[i, j]
            aug_e = aug_examples_matrix[i, j]
            if data_e is not None and aug_e is not None:
                data_label = data_e['is_close'][1]
                aug_label = aug_e['is_close'][1]
                if data_label == aug_label:
                    best_idx = i
                    break
        best_d = distances_for_data_j[best_idx]
        diversity = distance_to_score(best_d)
        diversities.append(diversity)
    return np.array(diversities)


def compute_plausibility(distances_matrix, aug_examples_matrix, data_examples_matrix):
    plausibilities = []
    for i in range(distances_matrix.shape[0]):
        distances_for_aug_i = distances_matrix[i]

        sorted_j = np.argsort(distances_for_aug_i)
        best_idx = sorted_j[0]
        for j in sorted_j:
            data_e = data_examples_matrix[i, j]
            aug_e = aug_examples_matrix[i, j]
            if data_e is not None and aug_e is not None:
                data_label = data_e['is_close'][1]
                aug_label = aug_e['is_close'][1]
                if data_label == aug_label:
                    best_idx = j
                    break

        best_d = distances_for_aug_i[best_idx]
        plausibility = distance_to_score(best_d)
        plausibilities.append(plausibility)
    return np.array(plausibilities)
