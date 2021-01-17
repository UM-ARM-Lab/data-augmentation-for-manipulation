import argparse
import re
import shutil
from enum import Enum


class ArgsEnum(Enum):

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return cls[s]
        except KeyError:
            raise ValueError()


def my_formatter(prog):
    size = shutil.get_terminal_size((80, 20))
    return argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=size.columns, width=size.columns)


def point_arg(i):
    try:
        x, y = [d.strip(" ") for d in i.split(",")]
        x = float(x)
        y = float(y)
        return x, y
    except Exception:
        raise ValueError("Failed to parse {} into two floats. Must be comma seperated".format(i))


def bool_arg(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_bool_arg(parser: argparse.ArgumentParser, flag: str, required: bool = True, help: str = ""):
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument('--' + flag, action='store_true', help=help)
    group.add_argument('--no-' + flag, action='store_true', help="NOT " + help)


def int_range_arg(v):
    """
    :param v: either a single int or a range like 3-8 (both ends inclusive)
    :return: list of ints
    """
    try:
        i = int(v)
        return [i]
    except ValueError:
        pass
    # parse things like 1-4
    m = re.fullmatch("(\d+)-(\d+)", v)
    try:
        if m:
            start = int(m.group(1))
            end = int(m.group(2))
            return list(range(start, end + 1))
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"invalid int range {v}")