import pathlib
from typing import List

import pandas as pd
from colorama import Fore
from tabulate import tabulate


class MyTable:
    def __init__(self, name: str, table_format: str, header: List[str]):
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

    def add_to_table(self, data: pd.DataFrame, series_name: str):
        x = data['x']
        self.table_data.append([series_name, x])

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


#     pvalue_table = dict_to_pvalue_table(figure.metric.values, table_format=table_format)
#     print(Style.BRIGHT + pvalue_table_title + Style.NORMAL)

class First(MyTable):

    def add_to_table(self, data: pd.DataFrame, series_name: str):
        x = data['x']
        success_start = x[x.first_valid_index()]
        success_end = x[x.last_valid_index()]
        self.table_data.append([series_name, success_start, success_end])


__all__ = [
    'MyTable',
    'First',
]
