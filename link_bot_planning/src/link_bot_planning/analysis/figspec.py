from dataclasses import dataclass
from typing import List, Dict

import pandas as pd

from link_bot_planning.analysis.results_figures import *
from link_bot_planning.analysis.results_tables import MyTable

DEFAULT_AXES_NAMES = ['x', 'y', 'z']


@dataclass
class FigSpec:
    fig: MyFigure
    reductions: Dict[str, List]
    axes_names: List[str]


@dataclass
class TableSpec:
    table: MyTable
    reductions: Dict[str, List]
    axes_names: List[str]


def get_data_for_figure(spec: FigSpec, metrics: pd.DataFrame):
    return reduce_metrics(spec.reductions, spec.axes_names, metrics)


def get_data_for_table(spec: TableSpec, metrics: pd.DataFrame):
    return reduce_metrics(spec.reductions, spec.axes_names, metrics)


def reduce_metrics(reductions: Dict[str, List], axis_names: List[str], metrics: pd.DataFrame):
    data_for_figure = None
    for axis_name, reductions_for_axis in reductions.items():
        data_for_axis = metrics[axis_name]
        index_names = list(data_for_axis.index.names)
        for reduction in reversed(reductions_for_axis):
            if isinstance(reduction, tuple):
                reduction, reduction_args = reduction
            else:
                reduction_args = tuple()
            index_names.pop(-1)
            if reduction is not None:
                data_for_axis_groupby = data_for_axis.groupby(index_names, group_keys=False)
                try:
                    data_for_axis = data_for_axis_groupby.agg(reduction, *reduction_args)
                    # NOTE: this handles the case where a function is specified in hjson, namely my_rolling
                except AttributeError:
                    data_for_axis = eval(reduction)(data_for_axis_groupby)
        if data_for_figure is None:
            data_for_figure = data_for_axis
        else:
            data_for_figure = pd.merge(data_for_figure, data_for_axis, on=metrics.index.names[:-1])

    # why is setting columns names so hard?
    # why does my code sometimes output a Series and sometimes a DataFrame?
    if isinstance(data_for_figure, pd.Series):
        data_for_figure = data_for_figure.to_frame()
        data_for_figure.columns = axis_names
    elif isinstance(data_for_figure, pd.DataFrame):
        columns = dict(zip(reductions.keys(), axis_names))
        data_for_figure.rename(columns=columns, inplace=True)
    else:
        raise NotImplementedError()

    return data_for_figure
