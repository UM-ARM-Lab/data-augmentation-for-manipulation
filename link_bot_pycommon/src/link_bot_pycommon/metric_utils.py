from typing import Dict, Optional

import numpy as np
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


def dict_to_pvalue_table(data_dict: Dict,
                         table_format: str = 'simple_grid',
                         fmt: str = '{:5.3f}',
                         title: Optional[str] = ''):
    """
    uses a one-sided T-test
    :param data_dict: A dictionary of method_name(str): values(list/array)
    :param table_format:
    :param fmt:
    :return:
    """
    pvalues = np.zeros([len(data_dict), len(data_dict) + 1], dtype=object)
    for i, (name1, e1) in enumerate(data_dict.items()):
        pvalues[i, 0] = name1
        for j, (_, e2) in enumerate(data_dict.items()):
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
    headers = [title] + list(data_dict.keys())
    table = tabulate(pvalues, headers=headers, tablefmt=table_format)
    return table
