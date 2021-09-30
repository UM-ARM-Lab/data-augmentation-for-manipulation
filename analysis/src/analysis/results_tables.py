import pathlib
import re
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from colorama import Fore, Style
from tabulate import tabulate

from link_bot_pycommon.metric_utils import dict_to_pvalue_table


def remove_uninformative_parts_of_paths(s: str):
    prefixes = [
        '/',
        'media',
        'shared',
        'ift',
        'planning_results',
        'classifier_data',
        'cl_trials',
    ]
    suffixes = [
        '/',
        'best_checkpoint',
    ]
    no_prefixes_left = False
    while not no_prefixes_left:
        no_prefixes_left = True
        for prefix in prefixes:
            if s.startswith(prefix):
                no_prefixes_left = False
                s = s.lstrip(prefix)
    no_suffixes_left = False
    while not no_suffixes_left:
        no_suffixes_left = True
        for suffix in suffixes:
            if s.startswith(suffix):
                no_suffixes_left = False
                s = s.rstrip(suffix)
    return s


def remove_long_numbers(s: str):
    return re.sub(r'\d\d\d\d+', '', s)


def fix_long_string(s: str):
    s = remove_uninformative_parts_of_paths(s)
    s = remove_long_numbers(s)
    return s


def fix_long_strings(row):
    fixed = []
    for e in row:
        if isinstance(e, str):
            e = fix_long_string(e)
        fixed.append(e)
    return fixed