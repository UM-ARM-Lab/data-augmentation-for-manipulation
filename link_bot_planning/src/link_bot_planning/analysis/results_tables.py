import pathlib

import pandas as pd
from colorama import Fore
from tabulate import tabulate

from link_bot_pycommon.metric_utils import dict_to_pvalue_table


class MyTable:
    def __init__(self, name: str, table_format: str, header):
        super().__init__()
        self.table_data = []
        self.table_format = table_format
        self.name = name
        self.table = None
        self.header = header

    def make_table(self, data, series_names):
        # Methods need to have consistent colors across different plots
        for series_name in series_names:
            data_for_series = data.loc[series_name]
            self.add_to_table(data_for_series, series_name=series_name)

        self.table = tabulate(self.table_data,
                              headers=self.header,
                              tablefmt=self.table_format,
                              floatfmt='6.4f',
                              numalign='center',
                              stralign='left')

    def add_to_table(self, data: pd.Series, series_name: str):
        self.table_data.append([series_name] + data.to_list())

    def save(self, outdir: pathlib.Path):
        filename = outdir / (self.name + ".txt")
        print(Fore.GREEN + f"Saving {filename}")

        # For saving metrics since this script is kind of slow it's nice to save the output
        with filename.open("w") as tables_file:
            tables_file.write(self.name)
            tables_file.write('\n')
            tables_file.write(self.table)
            tables_file.write('\n')

    def print(self):
        print(self.table)


class PValuesTable(MyTable):

    def __init__(self, name: str, table_format: str):
        super().__init__(name, table_format, None)
        self.table_data = []
        self.table_format = table_format
        self.name = name
        self.table = None

    def make_table(self, data, series_names):
        arrays_per_method = {}
        for series_name in series_names:
            data_for_series = data.loc[series_name]
            x = data_for_series['x']
            arrays_per_method[series_name] = x

        self.table = dict_to_pvalue_table(arrays_per_method, table_format=self.table_format, title=self.name)


__all__ = [
    'MyTable',
    'PValuesTable',
]
