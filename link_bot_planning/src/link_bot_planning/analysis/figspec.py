from dataclasses import dataclass, field
from typing import List, Dict

import pandas as pd

from link_bot_planning.analysis.results_figures import MyFigure


@dataclass
class FigSpec:
    fig: MyFigure
    reductions: Dict[str, List[str]]
    axis_names: List[str] = field(default_factory=lambda: ['x', 'y', 'z'])


def get_data_for_figure(spec: FigSpec, metrics: pd.DataFrame):
    data_for_figure = None
    for axis_name, reductions in spec.reductions.items():
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

    # why is setting columns names so hard?
    # why does my code sometimes output a Series and sometimes a DataFrame?
    if isinstance(data_for_figure, pd.Series):
        data_for_figure = data_for_figure.to_frame()
        data_for_figure.columns = spec.axis_names
    else:
        columns = dict(zip(spec.reductions.keys(), spec.axis_names))
        data_for_figure.rename(columns=columns, inplace=True)

    return data_for_figure
