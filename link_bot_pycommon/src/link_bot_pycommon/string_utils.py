from typing import List

from IPython.utils.text import long_substr


def shorten(name):
    return name.replace('-', '').replace('_', '').replace('/', '').replace(' ', '')


def make_useful_names(names: List[str]):
    useful_names = names.copy()
    while True:
        s = long_substr(useful_names)
        s = s.lstrip('-_')
        if len(s) <= 1:
            break
        useful_names = [n.replace(s, '') for n in useful_names]
    useful_names = [n.strip('-_') for n in useful_names]
    return useful_names