import pathlib
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.latex_utils import make_cell
from link_bot_pycommon.matplotlib_utils import save_unconstrained_layout
from link_bot_pycommon.metric_utils import row_stats


class ResultsMetric:
    def __init__(self, args, results_dir: pathlib.Path):
        super().__init__()
        self.args = args
        self.results_dir = results_dir
        self.values = {}
        self.method_indices = {}

    def setup_method(self, method_name: str, metadata: Dict):
        self.values[method_name] = []

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        raise NotImplementedError()

    def aggregate_trial(self, method_name: str, scenario: ExperimentScenario, trial_datum: Dict):
        metric_value = self.get_metric(scenario, trial_datum)
        self.values[method_name].append(metric_value)

    def aggregate_metric_values(self, method_name: str, metric_values):
        self.values[method_name].append(metric_values)

    def convert_to_numpy_arrays(self):
        for method_name, metric_values in self.values.items():
            self.values[method_name] = np.array(metric_values)


class BoxplotOverTrialsPerMethod(ResultsMetric):
    def __init__(self, args, results_dir: pathlib.Path):
        super().__init__(args, results_dir)

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        return trial_datum['total_time']


class TaskError(ResultsMetric):
    def __init__(self, args, results_dir: pathlib.Path):
        super().__init__(args, results_dir)
        self.goal_threshold = None

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        goal = trial_datum['goal']
        final_actual_state = trial_datum['end_state']
        final_execution_to_goal_error = scenario.distance_to_goal(final_actual_state, goal)
        return final_execution_to_goal_error

    def setup_method(self, method_name: str, metadata: Dict):
        super().setup_method(method_name, metadata)
        planner_params = metadata['planner_params']
        if 'goal_params' in planner_params:
            self.goal_threshold = planner_params['goal_params']['threshold']
        else:
            self.goal_threshold = planner_params['goal_threshold']


class NRecoveryActions(BoxplotOverTrialsPerMethod):
    def __init__(self, args, results_dir: pathlib.Path):
        super().__init__(args, results_dir)

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        steps = trial_datum['steps']
        n_recovery = 0
        for step in steps:
            if step['type'] == 'executed_recovery':
                n_recovery += 1
        return n_recovery


class NPlanningAttempts(BoxplotOverTrialsPerMethod):
    def __init__(self, args, results_dir: pathlib.Path):
        super().__init__(args, results_dir)

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        return len(trial_datum['steps'])


class TotalTime(BoxplotOverTrialsPerMethod):
    def __init__(self, args, results_dir: pathlib.Path):
        super().__init__(args, results_dir)

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        return trial_datum['total_time']


class MyFigure:
    def __init__(self, analysis_params: Dict, metric: ResultsMetric, name: str):
        super().__init__()
        self.metric = metric
        self.params = analysis_params
        self.name = name
        self.fig, self.ax = self.create_figure()

    def create_figure(self):
        raise NotImplementedError()

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
            colors = self.params["colors"]
            color = colors.get(method_name, None)
            self.add_to_figure(method_name=method_name, values=values_for_method, color=color)
        self.finish_figure()

    def add_to_figure(self, method_name: str, values: List, color):
        raise NotImplementedError()

    def finish_figure(self):
        self.ax.legend()

    def save_figure(self):
        filename = self.metric.results_dir / (self.name + ".jpeg")
        print(Fore.GREEN + f"Saving {filename}")
        save_unconstrained_layout(self.fig, filename, dpi=300)

    def enumerate_methods(self):
        for i, k in enumerate(self.metric.values):
            self.metric.method_indices[k] = i

    def sort_methods(self, sort_order: Dict):
        sorted_values = {k: self.metric.values[k] for k in sort_order.keys()}
        self.metric.values = sorted_values
        self.enumerate_methods()


class BoxplotOverTrialsPerMethodFigure(MyFigure):
    def __init__(self, analysis_params: Dict, metric, ylabel: str):
        super().__init__(analysis_params, metric, name="task_error_boxplot")
        self.ax.set_xlabel("Method")
        self.ax.set_ylabel(ylabel)
        self.trendline = self.params.get('trendline', False)

    def create_figure(self):
        return plt.subplots(figsize=(7.3, 4))

    def add_to_figure(self, method_name: str, values: List, color):
        x = self.metric.method_indices[method_name]
        if self.trendline:
            self.ax.plot(x, np.mean(values, axis=0), c=color, zorder=2)
        self.ax.boxplot(values,
                        positions=[x],
                        widths=0.9,
                        patch_artist=True,
                        boxprops=dict(facecolor='#00000000', color=color),
                        capprops=dict(color=color),
                        whiskerprops=dict(color=color),
                        medianprops=dict(color=color),
                        showfliers=False)
        plt.setp(self.ax.get_xticklabels(), rotation=18, horizontalalignment='right')

    def get_table_header(self):
        return ["Name", self.name]

    def finish_figure(self):
        # don't a legend for these plots
        self.ax.set_xticklabels(self.metric.values.keys())


class TaskErrorLineFigure(MyFigure):
    def __init__(self, analysis_params: Dict, metric: TaskError):
        super().__init__(analysis_params, metric, name="task_error_lineplot")
        self.fig.suptitle(self.params['experiment_name'])
        max_error = self.params["max_error"]
        self.errors_thresholds = np.linspace(0.01, max_error, self.params["n_error_bins"])
        self.ax.set_xlabel("Task Error Threshold (m)")
        self.ax.set_ylabel("Success Rate")
        self.ax.set_ylim([-0.1, 100.5])
        self.fig.subplots_adjust(top=0.94)

    def create_figure(self):
        return plt.subplots(figsize=(7.3, 4))

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


class NRecoveryActionsFigure(BoxplotOverTrialsPerMethodFigure):
    def __init__(self, analysis_params: Dict, metric):
        super().__init__(analysis_params, metric, "Recovery Actions")


class NPlanningAttemptsFigure(BoxplotOverTrialsPerMethodFigure):
    def __init__(self, analysis_params: Dict, metric):
        super().__init__(analysis_params, metric, "Planning Attempts")


class TotalTimeBoxplotFigure(BoxplotOverTrialsPerMethodFigure):
    def __init__(self, analysis_params: Dict, metric):
        super().__init__(analysis_params, metric, "Total Time")


class TaskErrorBoxplotFigure(BoxplotOverTrialsPerMethodFigure):
    def __init__(self, analysis_params: Dict, metric):
        super().__init__(analysis_params, metric, "Task Error")
        self.ax.set_ylim([0.0, self.params["max_error"]])
        self.fig.suptitle(self.params['experiment_name'])

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        goal = trial_datum['goal']
        final_actual_state = trial_datum['end_state']
        self.thresholds = trial_datum['planner_params']
        final_execution_to_goal_error = scenario.distance_to_goal(final_actual_state, goal)
        return final_execution_to_goal_error

    def add_to_figure(self, method_name: str, values: List, color):
        super().add_to_figure(method_name, values, color)
        x = self.metric.method_indices[method_name]
        n_values = len(values)
        xs = [x] * n_values + np.random.RandomState(0).uniform(-0.08, 0.08, size=n_values)
        self.ax.scatter(xs, values, edgecolors='k', s=5, marker='o', facecolors='none')

    def finish_figure(self):
        values = np.array(list(self.metric.values.values()))
        if values.ndim < 2:
            return

        self.ax.plot(range(len(values)), np.mean(values, axis=1), c='b', zorder=2)
        self.ax.set_xticklabels(list(self.metric.values.keys()))

    def get_table_header(self):
        return ["Name", "min", "max", "mean", "median", "std"]
