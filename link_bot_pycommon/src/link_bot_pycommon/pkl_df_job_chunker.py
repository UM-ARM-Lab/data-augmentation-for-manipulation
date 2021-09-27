import operator
import pathlib
import pickle
from functools import reduce
from typing import Dict

import pandas as pd


class DfJobChunker:

    def __init__(self, logfile_name: pathlib.Path):
        self.logfile_name = logfile_name
        self.df = pd.DataFrame()
        self.load()

    def load(self):
        if self.logfile_name.exists():
            with self.logfile_name.open("rb") as df_file:
                self.df = pickle.load(df_file)

    def save(self):
        with self.logfile_name.open("wb") as df_file:
            pickle.dump(self.df, df_file)

    def append(self, row: Dict, verify_integrity=True):
        self.df = self.df.append(row, verify_integrity=verify_integrity, ignore_index=True)
        self.save()

    def has(self, row: Dict):
        if self.df.size == 0:
            return False
        conditions = []
        for k, v in row.items():
            if k not in self.df.columns:
                return False
            conditions.append(self.df[k] == v)
        return reduce(operator.iand, conditions).any()
