import pathlib
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from colorama import Fore, Style
from tabulate import tabulate

from link_bot_pycommon.metric_utils import dict_to_pvalue_table


def fix_long_string(s: str):
    prefixes = [
        '/',
        'media',
        'shared',
        'ift',
        'planning_results',
        'classifier_data',
    ]
    no_prefixes_left = False
    while not no_prefixes_left:
        no_prefixes_left = True
        for prefix in prefixes:
            if s.startswith(prefix):
                no_prefixes_left = False
                s = s.lstrip(prefix)
    return s


def fix_long_strings(row: Iterable):
    return [fix_long_string(str(e)) for e in row]


class MyTable:
    def __init__(self, name: str, table_format: str, header):
        super().__init__()
        self.table_data = []
        self.table_format = table_format
        self.name = name
        self.table = None
        self.header = header

    def make_table(self, data):
        # Methods need to have consistent colors across different plots
        for i, s in data.iterrows():
            self.add_to_table(s, series_name=i)

        self.table = tabulate(self.table_data,
                              headers=self.header,
                              tablefmt=self.table_format,
                              floatfmt='6.4f',
                              numalign='center',
                              stralign='left')

    def add_to_table(self, data: pd.Series, series_name: str):
        row = data.to_list()
        row = fix_long_strings(row)
        self.table_data.append(row)

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
        print(Style.BRIGHT + Fore.LIGHTYELLOW_EX + self.name + Style.RESET_ALL)
        print(self.table)


class PValuesTable(MyTable):

    def __init__(self, name: str, table_format: str):
        super().__init__(name, table_format, None)
        self.table_format = table_format
        self.name = name
        self.table = None

    def make_table(self, data: pd.DataFrame):
        arrays_per_method = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            indices = data.index.unique()

        indices_matrix = np.array([list(v) for v in indices.values])
        useful_levels = np.all(indices_matrix[0] == indices_matrix, axis=0)
        useful_levels_indices = np.where(np.logical_not(useful_levels))[0]
        if np.size(useful_levels_indices) == 0:
            useful_level_names = indices.names
        else:
            useful_level_names = np.take(indices.names, useful_levels_indices).tolist()

        data.index = pd.MultiIndex.from_frame(data.index.to_frame().fillna('na'))
        for index, values in data.groupby(useful_level_names, dropna=False):
            if isinstance(index, str):
                name = index
            else:
                name = '-'.join(fix_long_strings(index))
            arrays_per_method[name] = values.squeeze()

        self.table = dict_to_pvalue_table(arrays_per_method, table_format=self.table_format, title=self.name)


__all__ = [
    'MyTable',
    'PValuesTable',
]
