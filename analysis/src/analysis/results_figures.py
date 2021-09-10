import pathlib
import seaborn as sns
import re
from typing import Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import Fore

from link_bot_pycommon.matplotlib_utils import save_unconstrained_layout, adjust_lightness, get_rotation
from link_bot_pycommon.pandas_utils import rlast

colors_cache = {}


class MyFigure:
    def __init__(self, analysis_params: Dict, name: str):
        super().__init__()
        self.params = analysis_params
        self.name = name
        self.fig, self.ax = self.create_figure()

    def create_figure(self):
        fig, ax = plt.subplots(figsize=self.get_figsize())
        return fig, ax

    def make_figure(self, data, series_names):
        # Methods need to have consistent colors across different plots
        for series_idx, series_name in enumerate(series_names):
            color = self.get_color_for_method(series_name)
            data_for_series = data.loc[series_name]
            self.add_to_figure(data_for_series, series_name=series_name, color=color, series_idx=series_idx)
        self.finish_figure()

    def get_color_for_method(self, method_name):
        colors = self.params["colors"]
        method_name_for_color = method_name.replace("*", "")

        for color_pattern, color in colors.items():
            if re.fullmatch(color_pattern, method_name_for_color):
                return color

        m = re.fullmatch(r"(.*?) \((\d+)\)", method_name_for_color)
        if m:
            method_name_without_number = m.group(0)
            if method_name_without_number in colors:
                color = colors[method_name_without_number]
                return color

        color = colors_cache.get(method_name, None)
        if color is not None:
            return color

        print(Fore.YELLOW + f"color is None! choosing a random color for method {method_name}")
        color = np.random.uniform(0, 1, 3)
        colors_cache[method_name] = color
        return color

    def add_to_figure(self, data: List, series_name: str, color, series_idx: int):
        raise NotImplementedError()

    def finish_figure(self):
        self.ax.legend()

    def methods_on_x_axis(self):
        xticklabels = list(self.metric.values.keys())
        self.ax.set_xticks(list(self.metric.method_indices.values()))
        self.ax.set_xticklabels(xticklabels)
        plt.setp(self.ax.get_xticklabels(), rotation=get_rotation(xticklabels=xticklabels), horizontalalignment='right')

    def save_figure(self, outdir: pathlib.Path):
        filename = outdir / (self.name + ".jpeg")
        print(Fore.GREEN + f"Saving {filename}")
        save_unconstrained_layout(self.fig, filename, dpi=300)

    def enumerate_methods(self, metric):
        for i, k in enumerate(metric.values):
            metric.method_indices[k] = i

    def get_figsize(self):
        #     return get_figsize(len(self.metric.values))
        return 10, 5


class LinePlot(MyFigure):
    def __init__(self, analysis_params: Dict, name: str, xlabel: str, ylabel: str, ylim=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        super().__init__(analysis_params, name)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        if ylim is not None:
            self.ax.set_ylim(ylim)

    def add_to_figure(self, data: pd.DataFrame, series_name: str, color, series_idx: int):
        if 'y' in data:
            y = data['y'].values
            x = data['x'].values
            self.ax.plot(x, y, c=color, label=series_name)
        else:
            y = data['x'].values
            self.ax.plot(y, c=color, label=series_name)


def try_set_violinplot_color(parts, key, color):
    if key in parts:
        parts[key].set_edgecolor(color)


class ViolinPlot(MyFigure):
    def __init__(self, analysis_params: Dict, name: str, xlabel: str, ylabel: str, ylim=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        super().__init__(analysis_params, name)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        if ylim is not None:
            self.ax.set_ylim(ylim)
        self.handles = []
        self.series_names = []

    def add_to_figure(self, data: pd.DataFrame, series_name: str, color, series_idx: int):
        assert 'y' not in data
        y = data['x'].values
        x = [series_idx]
        parts = self.ax.violinplot(y,
                                   positions=x,
                                   showmeans=True,
                                   showextrema=False,
                                   showmedians=True,
                                   widths=0.9)

        try_set_violinplot_color(parts, 'cmeans', adjust_lightness(color, 0.8))
        try_set_violinplot_color(parts, 'cmedians', adjust_lightness(color, 1.2))
        try_set_violinplot_color(parts, 'cmaxes', 'k')
        try_set_violinplot_color(parts, 'cmins', 'k')
        try_set_violinplot_color(parts, 'cbars', 'k')
        if 'bodies' in parts:
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_edgecolor(color)

        handle = mpatches.Patch(color=color)
        self.handles.append(handle)
        self.series_names.append(series_name)

    def finish_figure(self):
        super().finish_figure()
        self.ax.legend(self.handles, self.series_names)


class BoxPlot(MyFigure):
    def __init__(self, analysis_params: Dict, name: str, xlabel: str, ylabel: str, ylim=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        super().__init__(analysis_params, name)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        if ylim is not None:
            self.ax.set_ylim(ylim)

    def add_to_figure(self, data: pd.DataFrame, series_name: str, color, series_idx: int):
        assert 'y' not in data
        y = data['x'].values
        x = [series_idx]
        self.ax.boxplot(y,
                        positions=x,
                        widths=0.9,
                        patch_artist=True,
                        boxprops=dict(facecolor='#00000000', color=color),
                        capprops=dict(color=color),
                        whiskerprops=dict(color=color),
                        medianprops=dict(color=color),
                        showfliers=False,
                        )
        x_repeated = [series_idx] * len(y) + np.random.randn(*y.shape) * 0.05
        self.ax.scatter(x_repeated, y, edgecolor=color, facecolor='none', marker='o', label=series_name)


class BarChart(MyFigure):
    def __init__(self, analysis_params: Dict, name: str, xlabel: str, ylabel: str, ylim=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        super().__init__(analysis_params, name)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        if ylim is not None:
            self.ax.set_ylim(ylim)

    def add_to_figure(self, data: pd.DataFrame, series_name: str, color, series_idx: int):
        assert 'y' not in data
        y = data['x']
        x = [series_idx]
        self.ax.bar(x, y, color=color, label=series_name)


class LineBoxPlot(LinePlot):

    def add_to_figure(self, data: pd.DataFrame, series_name: str, color, series_idx: int):
        super().add_to_figure(data, series_name, color, series_idx)
        y = data['y'].values
        if 'x' in data:
            x = data['x'].values
        else:
            x = list(range(len(y)))
        self.ax.boxplot(y,
                        positions=x,
                        widths=0.9,
                        patch_artist=True,
                        boxprops=dict(facecolor='#00000000', color=color),
                        capprops=dict(color=color),
                        whiskerprops=dict(color=color),
                        medianprops=dict(color=color),
                        showfliers=False)


def my_rolling(window: int = 10):
    return lambda x: x.rolling(window=window, min_periods=0).mean()


def shifted_cumsum(x):
    return x.cumsum() - x.first()


def lineplot(df,
             x: str,
             metric: str,
             title: str,
             window: int = 1,
             hue: Optional[str] = None):
    agg = {metric: 'mean', x: rlast}
    if hue is not None:
        z = df.groupby(hue).rolling(window).agg(agg)
    else:
        z = df.rolling(window).agg(agg)
    plt.figure()
    ax = sns.lineplot(
        data=z,
        x=x,
        y=metric,
        hue=hue,
        palette='colorblind',
        estimator='mean',
        ci=80,
    )
    ax.set_title(title)
    return ax


__all__ = [
    'BarChart',
    'BoxPlot',
    'LineBoxPlot',
    'LinePlot',
    'MyFigure',
    'ViolinPlot',
    'my_rolling',
    'shifted_cumsum',
]
