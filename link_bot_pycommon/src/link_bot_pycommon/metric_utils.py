from typing import Dict, Optional, Generator

import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate


def make_row(metric_name, metric_data):
    row = [metric_name]
    row.extend(row_stats(metric_data))
    return row


def row_stats(metric_data):
    return [np.min(metric_data), np.max(metric_data), np.mean(metric_data), np.median(metric_data), np.std(metric_data)]


def brief_row_stats(metric_data):
    return [np.mean(metric_data), np.median(metric_data), np.std(metric_data)]


def dict_to_pvalue_table(data_dict: Dict, **kwargs):
    """ uses a one-sided T-test """

    def gen():
        for i, (name1, e1) in enumerate(data_dict.items()):
            for j, (name2, e2) in enumerate(data_dict.items()):
                yield i, j, name1, e1, name2, e2

    return gen_to_pvalue_table(gen=gen(), n=len(data_dict), **kwargs)


def df_to_pvalue_table(df: pd.DataFrame, y: str, **kwargs):
    """ uses a one-sided T-test """

    groups = df.groupby('method_name')

    def gen():
        for i, (name1, group1) in enumerate(groups):
            e1 = group1[y].values
            for j, (name2, group2) in enumerate(groups):
                e2 = group2[y].values
                yield i, j, name1, e1, name2, e2

    return gen_to_pvalue_table(gen=gen(), n=len(groups), **kwargs)


def gen_to_pvalue_table(gen: Generator,
                        n: int,
                        table_format: str = 'simple_grid',
                        fmt: str = '{:5.4f}',
                        title: Optional[str] = ''):
    """ uses a one-sided T-test """
    pvalues = np.zeros([n, n + 1], dtype=object)
    for i, j, name1, e1, name2, e2 in gen:
        pvalues[i, 0] = name1
        _, pvalue = stats.ttest_ind(e1, e2)
        # one-sided, we simply divide pvalue by 2
        pvalue = pvalue / 2
        if pvalue < 0.01:
            prefix = "! "
        else:
            prefix = "  "
        if j != i:
            pvalues[i, j + 1] = prefix + fmt.format(pvalue)
        else:
            pvalues[i, j + 1] = '-'

    names = [gen_i[2] for gen_i in gen]
    names = np.unique(names).tolist()
    headers = [title] + names
    table = tabulate(pvalues, headers=headers, tablefmt=table_format)
    return table
