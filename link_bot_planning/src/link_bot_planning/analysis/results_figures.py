import pathlib
import re
from typing import Dict, List, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import Fore

from link_bot_pycommon.matplotlib_utils import save_unconstrained_layout, adjust_lightness, get_rotation

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
        for series_name in series_names:
            color = self.get_color_for_method(series_name)
            data_for_series = data.loc[series_name]
            self.add_to_figure(data_for_series, series_name=series_name, color=color)
        self.finish_figure()

    def get_color_for_method(self, method_name):
        colors = self.params["colors"]
        method_name_for_color = method_name.replace("*", "")

        if method_name[-3:] == " ft":
            base_method_name = method_name[:-3]
            color = self.get_color_for_method(base_method_name)
            color = adjust_lightness(color, 0.8)
            return color

        if method_name.split(" ")[-1] == "classifier":
            base_method_name = " ".join(method_name.split(" ")[:-1])
            color = self.get_color_for_method(base_method_name)
            color = adjust_lightness(color, 1.2)
            return color

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

    def add_to_figure(self, data: List, series_name: str, color):
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

    # def sort_methods(self, metric, sort_order: Dict):
    #     sorted_values = {k: metric.values[k] for k in sort_order.keys()}
    #     metric.values = sorted_values
    #     self.enumerate_methods()

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

    def add_to_figure(self, data: pd.DataFrame, series_name: str, color):
        y = data['y'].values
        if 'x' in data:
            x = data['x'].values
            self.ax.plot(x, y, c=color, label=series_name)
        else:
            self.ax.plot(y, c=color, label=series_name)


class LineBoxPlot(LinePlot):

    def add_to_figure(self, data: pd.DataFrame, series_name: str, color):
        super().add_to_figure(data, series_name, color)
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


def make_figures(figures: Iterable[MyFigure],
                 analysis_params: Dict,
                 sort_order_dict: Dict,
                 out_dir: pathlib.Path):
    for figure in figures:
        figure.params = analysis_params
        figure.sort_methods(sort_order_dict)

    for figure in figures:
        figure.enumerate_methods()

    # Actual figures
    for figure in figures:
        figure.make_figure()
        figure.save_figure(out_dir)


def my_rolling(window: int = 10):
    return lambda x: x.rolling(window=window, min_periods=0).mean()


def shifted_cumsum(x):
    return x.cumsum() - x.first()


__all__ = [
    'my_rolling',
    'shifted_cumsum',
    'MyFigure',
    'LinePlot',
    'LineBoxPlot',
]
