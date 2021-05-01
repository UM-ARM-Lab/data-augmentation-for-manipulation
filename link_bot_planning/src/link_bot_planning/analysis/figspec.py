from dataclasses import dataclass
from typing import List

import numpy as np

from link_bot_planning.analysis.results_figures import MyFigure
from link_bot_planning.analysis.results_metrics import Reduction


@dataclass
class FigSpec:
    fig: MyFigure
    metrics_indices: List
    metrics: List
    reductions: List[List[Reduction]]


def make_groups(metrics_indices: np.ndarray, metrics: np.ndarray, n_matching_dims: int):
    groups_indices = []
    groups_values = []
    group_indices = []
    group_value = []
    indices_to_match = metrics_indices[0][:n_matching_dims]
    for indices_i, value_i in zip(metrics_indices, metrics):
        indices_to_match_i = indices_i[:n_matching_dims]
        if np.all(indices_to_match_i == indices_to_match):
            group_indices.append(indices_i)
            group_value.append(value_i)
        else:
            indices_to_match = indices_to_match_i
            groups_indices.append(group_indices)
            groups_values.append(group_value)
            group_indices = [indices_i]
            group_value = [value_i]

    groups_indices.append(group_indices)
    groups_values.append(group_value)

    return groups_indices, groups_values


def restructure(metric_indices, metric_values):
    # by_method = []
    # lists_for_method = [x_metric_where_method is 0, y]
    # by_method.append(lists_for_method)
    # for metric_indices_i, metric_values_i in zip(metric_indices, metric_values):
    #     for metric_indices_i_j, metric_values_i_j in zip(metric_indices_i, metric_values_i):
    #         if metric_indices_i_j[0] == method_idx
    #             out_i.append(metric_values_i_j)
    #
    # return by_method
    pass


def reduce_metrics_for_figure(figspec: FigSpec):
    reduced_metric_indices = []
    reduced_metric = []
    for metric_indices, metric, reductions in zip(figspec.metrics_indices, figspec.metrics, figspec.reductions):
        assert len(reductions) <= len(metric_indices[0])
        for reduction_idx, reduction in enumerate(reversed(reductions)):
            # data is a 2d array, each row looks like [i, j, k, ..., data]
            n_matching_dims = len(metric_indices[0]) - 1
            groups_indices, groups_values = make_groups(metric_indices, metric, n_matching_dims)
            reduced_groups_indices = []
            reduced_groups_values = []
            for group_indices, group_values in zip(groups_indices, groups_values):
                group_indices = np.array(group_indices)
                group_values = np.array(group_values)
                if reduction is not None:
                    reduced_group_value = reduction(group_values, axis=axis)
                    if reduction.reduces:
                        reduced_group_indices = group_indices[0, :-1]  # -2 to drop the last index
                    else:
                        reduced_group_indices = group_indices
                else:
                    reduced_group_indices = group_indices
                    reduced_group_value = group_values

                if reduction.reduces:
                    reduced_groups_indices.append(reduced_group_indices)
                    reduced_groups_values.append(reduced_group_value)
                else:
                    reduced_groups_indices.extend(reduced_group_indices)
                    reduced_groups_values.extend(reduced_group_value)
            metric_indices = np.array(reduced_groups_indices)
            metric = np.array(reduced_groups_values)

        reduced_metric_indices.append(metric_indices)
        reduced_metric.append(metric)

    return reduced_metric_indices, reduced_metric
