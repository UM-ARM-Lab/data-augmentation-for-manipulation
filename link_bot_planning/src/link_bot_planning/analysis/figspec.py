from dataclasses import dataclass
from typing import List, Dict

import pandas as pd

from link_bot_planning.analysis.results_figures import MyFigure


@dataclass
class FigSpec:
    fig: MyFigure
    reductions: Dict[str, List[str]]


def get_data_for_figure(figspec: FigSpec, metrics: pd.DataFrame, axis_names=None):
    if axis_names is None:
        axis_names = ['x', 'y', 'z']

    data_for_figure = None
    for axis_name, reductions in figspec.reductions.items():
        data_for_axis = metrics[axis_name]
        index_names = list(data_for_axis.index.names)
        for reduction in reversed(reductions):
            index_names.pop(-1)
            if reduction is not None:
                data_for_axis = data_for_axis.groupby(index_names)
                data_for_axis = data_for_axis.agg(reduction)
        if data_for_figure is None:
            data_for_figure = data_for_axis
        else:
            data_for_figure = pd.merge(data_for_figure, data_for_axis, on=metrics.index.names[:-1])

    columns = dict(zip(figspec.reductions.keys(), axis_names))
    data_for_figure.rename(columns=columns, inplace=True)

    return data_for_figure
