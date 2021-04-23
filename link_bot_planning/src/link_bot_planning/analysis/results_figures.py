import pathlib
import re
from typing import Dict, List, Iterable

import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore
from matplotlib.lines import Line2D

from link_bot_data.visualization import color_violinplot
from link_bot_planning.analysis.results_metrics import TrialMetrics
from link_bot_planning.analysis.results_utils import classifer_dataset_params_from_planner_params
from link_bot_pycommon.latex_utils import make_cell
from link_bot_pycommon.matplotlib_utils import save_unconstrained_layout, adjust_lightness, get_rotation, get_figsize
from link_bot_pycommon.metric_utils import row_stats
from link_bot_pycommon.pycommon import quote_string
import pandas as pd

colors_cache = {}


class MyFigure:
    def __init__(self, analysis_params: Dict, metric: TrialMetrics, name: str):
        super().__init__()
        self.metric = metric
        self.params = analysis_params
        self.name = name
        self.fig, self.ax = self.create_figure()

    def set_title(self):
        metadata_for_method = next(iter(self.metric.metadatas.values()))
        cl_data_params = classifer_dataset_params_from_planner_params(metadata_for_method['planner_params'])
        classifier_training_scene_name_for_method = cl_data_params['data_collection_params']['name']
        # noinspection PyUnusedLocal
        classifier_training_scene = classifier_training_scene_name_for_method
        # noinspection PyUnusedLocal
        scene_name = metadata_for_method['scene_name']

        # This magic lets us specify variable names in the title in the json file. Any variable name used there
        # must be in scope here, hence the above local variables which appear unused
        title = eval(quote_string(self.params['experiment_name']))

        self.fig.suptitle(title)

    def create_figure(self):
        fig, ax = plt.subplots(figsize=self.get_figsize())
        return fig, ax

    def make_table(self, table_format):
        table_data = []
        for method_name, values_for_method in self.metric.values.items():
            table_data.append(self.make_row(method_name, values_for_method, table_format))
        return self.get_table_header(), table_data

    def get_table_header(self):
        raise NotImplementedError()

    def make_row(self, method_name: str, values_for_method: np.array, table_format: str):
        row = [
            make_cell(method_name, table_format),
        ]
        row.extend(row_stats(values_for_method))
        return row

    def make_figure(self):
        # Methods need to have consistent colors across different plots
        for method_name, values_for_method in self.metric.values.items():
            color = self.get_color_for_method(method_name)
            self.add_to_figure(method_name=method_name, values=values_for_method, color=color)
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

        if method_name_for_color in colors:
            color = colors[method_name_for_color]
            return color
        else:
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

    def add_to_figure(self, method_name: str, values: List, color):
        raise NotImplementedError()

    def finish_figure(self):
        self.set_title()
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

    def enumerate_methods(self):
        for i, k in enumerate(self.metric.values):
            self.metric.method_indices[k] = i

    def sort_methods(self, sort_order: Dict):
        sorted_values = {k: self.metric.values[k] for k in sort_order.keys()}
        self.metric.values = sorted_values
        self.enumerate_methods()

    def get_figsize(self):
        return get_figsize(len(self.metric.values))


# class MultiIterationsFigure:
#     def __init__(self, analysis_params: Dict, metric: TrialMetrics, name: str):
#         super().__init__(analysis_params, metric, name)
#
#     def set_title(self):
#         pass


class LinePlotAcrossIterationsFigure(MyFigure):
    def __init__(self, analysis_params: Dict, metric, ylabel: str):
        self.ylabel = ylabel
        name = self.ylabel.lower().replace(" ", "_") + "_lineplot"
        super().__init__(analysis_params, metric, name)
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel(self.ylabel)

    def get_table_header(self):
        return ["Name", "min", "max", "mean", "median", "std"]

    def add_to_figure(self, method_name: str, values: List, color):
        self.ax.plot(self.metric.values[method_name], c=color, label=method_name)

    def set_title(self):
        self.fig.suptitle(f"{self.ylabel} over time")

    def finish_figure(self):
        super().finish_figure()


def moving_window_average(x, window: int):
    return pd.DataFrame(x).rolling(window).mean()


class MWALinePlotAcrossIterationsFigure(LinePlotAcrossIterationsFigure):
    def add_to_figure(self, method_name: str, values: List, color):
        values = moving_window_average(self.metric.values[method_name], 5)
        self.ax.plot(values, c=color, label=method_name)

    def get_figsize(self):
        return 10, 5

class ViolinPlotOverTrialsPerMethodFigure(MyFigure):
    def __init__(self, analysis_params: Dict, metric, ylabel: str):
        name = ylabel.lower().replace(" ", "_") + "_violinplot"
        super().__init__(analysis_params, metric, name=name)
        self.ax.set_xlabel("Method")
        self.ax.set_ylabel(ylabel)
        self.trendline = self.params.get('trendline', False)

    def add_to_figure(self, method_name: str, values: List, color):
        x = self.metric.method_indices[method_name]
        if self.trendline:
            self.ax.plot(x, np.mean(values, axis=0), c=color, zorder=2, label='mean')
        if color is None:
            print(Fore.YELLOW + f"color is None! Set a color in the analysis file for method {method_name}")
            color = 'k'
        parts = self.ax.violinplot(values, positions=[x], widths=0.9, showmeans=True, bw_method=0.3)
        color_violinplot(parts, color)

        x = self.metric.method_indices[method_name]
        n_values = len(values)
        xs = [x] * n_values + np.random.RandomState(0).uniform(-0.08, 0.08, size=n_values)
        self.ax.scatter(xs, values, edgecolors='k', s=50, marker='o', facecolors='none')

    def finish_figure(self):
        mean_line = [Line2D([0], [0], color='#dddddd', lw=2)]
        self.ax.legend(mean_line, ['mean'])

        self.methods_on_x_axis()

    def set_title(self):
        pass

    def get_table_header(self):
        return ["Name", "min", "max", "mean", "median", "std"]


class BarChartPercentagePerMethodFigure(MyFigure):
    def __init__(self, analysis_params: Dict, metric: TrialMetrics, ylabel: str):
        name = ylabel.lower().replace(" ", "_") + "_barchart"
        super().__init__(analysis_params, metric, name)
        self.ax.set_xlabel("Method")
        self.ax.set_ylabel(ylabel)
        self.ax.set_ylim([-0.1, 100.5])

    def add_to_figure(self, method_name: str, values: List, color):
        x = self.metric.method_indices[method_name]
        if color is None:
            print(Fore.YELLOW + f"color is None! Set a color in the analysis file for method {method_name}")
        percentage_solved = np.sum(values) / values.shape[0] * 100
        self.ax.bar(x, percentage_solved, color=color)

    def finish_figure(self):
        super().finish_figure()
        self.methods_on_x_axis()

    def make_row(self, method_name: str, values_for_method: np.array, table_format: str):
        percentage_solved = np.sum(values_for_method) / values_for_method.shape[0] * 100
        row = [make_cell(method_name, table_format), percentage_solved]
        return row

    def get_table_header(self):
        return ["Name", "Value"]


class BoxplotOverTrialsPerMethodFigure(MyFigure):
    def __init__(self, analysis_params: Dict, metric, ylabel: str):
        name = ylabel.lower().replace(" ", "_") + "_boxplot"
        super().__init__(analysis_params, metric, name)
        self.ax.set_xlabel("Method")
        self.ax.set_ylabel(ylabel)
        self.trendline = self.params.get('trendline', False)

    def add_to_figure(self, method_name: str, values: List, color):
        x = self.metric.method_indices[method_name]
        if self.trendline:
            self.ax.plot(x, np.mean(values, axis=0), c=color, zorder=2, label='mean')
        if color is None:
            print(Fore.YELLOW + f"color is None! Set a color in the analysis file for method {method_name}")
        self.ax.boxplot(values,
                        positions=[x],
                        widths=0.9,
                        patch_artist=True,
                        boxprops=dict(facecolor='#00000000', color=color),
                        capprops=dict(color=color),
                        whiskerprops=dict(color=color),
                        medianprops=dict(color=color, linewidth=8),
                        showfliers=False)

        n_values = len(values)
        xs = [x] * n_values + np.random.RandomState(0).uniform(-0.08, 0.08, size=n_values)
        self.ax.scatter(xs, values, edgecolors='k', s=50, marker='o', facecolors='none')

    def finish_figure(self):
        super().finish_figure()
        self.methods_on_x_axis()

    def get_table_header(self):
        return ["Name", "min", "max", "mean", "median", "std"]


class TaskErrorLineFigure(MyFigure):
    def __init__(self, analysis_params: Dict, metric: TrialMetrics):
        super().__init__(analysis_params, metric, name="task_error_lineplot")
        max_error = self.params["max_error"]
        self.errors_thresholds = np.linspace(0.01, max_error, self.params["n_error_bins"])
        self.ax.set_xlabel("Task Error Threshold (m)")
        self.ax.set_ylabel("Success Rate")
        self.ax.set_ylim([-0.1, 100.5])

    def add_to_figure(self, method_name: str, values: List, color):
        success_rate_at_thresholds = []
        for threshold in self.errors_thresholds:
            success_rate_at_threshold = np.count_nonzero(values < threshold) / len(values) * 100
            success_rate_at_thresholds.append(success_rate_at_threshold)
        self.ax.plot(self.errors_thresholds, success_rate_at_thresholds, label=method_name, color=color)
        self.ax.axvline(self.metric.goal_threshold, color='#aaaaaa', linestyle='--')

    def get_table_header(self):
        return ["Name", "min", "max", "mean", "median", "std"]

    def make_row(self, method_name: str, values_for_method: np.array, table_format: str):
        row = [
            make_cell(method_name, table_format),
            # make_cell(table_config["dynamics"], tablefmt),
            # make_cell(table_config["classifier"], tablefmt),
        ]
        row.extend(row_stats(values_for_method))
        return row

    def get_figsize(self):
        return 10, 5


def box_plot(analysis_params: Dict, metric: TrialMetrics, name: str):
    return BoxplotOverTrialsPerMethodFigure(analysis_params, metric, name)


def violin_plot(analysis_params: Dict, metric: TrialMetrics, name: str):
    return ViolinPlotOverTrialsPerMethodFigure(analysis_params, metric, name)


def make_figures(figures: Iterable[MyFigure],
                 analysis_params: Dict,
                 sort_order_dict: Dict,
                 table_format: str,
                 tables_filename: pathlib.Path,
                 out_dir: pathlib.Path):
    for figure in figures:
        figure.params = analysis_params
        figure.sort_methods(sort_order_dict)

    for figure in figures:
        figure.enumerate_methods()

    # Tables
    # for figure in figures:
    #     table_header, table_data = figure.make_table(table_format)
    #     if table_data is None:
    #         continue
    #     print(Style.BRIGHT + figure.name + Style.NORMAL)
    #     table = tabulate(table_data,
    #                      headers=table_header,
    #                      tablefmt=table_format,
    #                      floatfmt='6.4f',
    #                      numalign='center',
    #                      stralign='left')
    #     print(table)
    #     print()
    #
    #     # For saving metrics since this script is kind of slow it's nice to save the output
    #     with tables_filename.open("a") as tables_file:
    #         tables_file.write(figure.name)
    #         tables_file.write('\n')
    #         tables_file.write(table)
    #         tables_file.write('\n')
    #
    # for figure in figures:
    #     pvalue_table_title = f"p-value matrix [{figure.name}]"
    #     pvalue_table = dict_to_pvalue_table(figure.metric.values, table_format=table_format)
    #     print(Style.BRIGHT + pvalue_table_title + Style.NORMAL)
    #     print(pvalue_table)
    #     with tables_filename.open("a") as tables_file:
    #         tables_file.write(pvalue_table_title)
    #         tables_file.write('\n')
    #         tables_file.write(pvalue_table)
    #         tables_file.write('\n')

    # Actual figures
    for figure in figures:
        figure.make_figure()
        figure.save_figure(out_dir)


__all__ = [
    'MyFigure',
    'ViolinPlotOverTrialsPerMethodFigure',
    'BoxplotOverTrialsPerMethodFigure',
    'BarChartPercentagePerMethodFigure',
    'TaskErrorLineFigure',
    'LinePlotAcrossIterationsFigure',
    'MWALinePlotAcrossIterationsFigure',
    'box_plot',
    'violin_plot',
]
