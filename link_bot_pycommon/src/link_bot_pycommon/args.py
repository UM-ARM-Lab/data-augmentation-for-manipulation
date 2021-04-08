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


def int_set_arg(v):
    """
    :param v: either a single int, or a range like 3-8 (both ends inclusive), or a csv list of ints
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
    # parse things like 1,2,3,4
    try:
        ints = []
        for v_i in v.split(","):
            ints.append(int(v_i))
        return ints
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"invalid int set {v}")


class BooleanOptionalAction(argparse.Action):
    def __init__(self,
                 option_strings,
                 dest,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):

        _option_strings = []
        for option_string in option_strings:
            _option_strings.append(option_string)

            if option_string.startswith('--'):
                option_string = '--no-' + option_string[2:]
                _option_strings.append(option_string)

        if help is not None and default is not None:
            help += f" (default: {default})"

        super().__init__(option_strings=_option_strings,
                         dest=dest,
                         nargs=0,
                         default=default,
                         type=type,
                         choices=choices,
                         required=required,
                         help=help,
                         metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.option_strings:
            setattr(namespace, self.dest, not option_string.startswith('--no-'))

    def format_usage(self):
        return ' | '.join(self.option_strings)


class BooleanAction(BooleanOptionalAction):
    def __init__(self,
                 option_strings,
                 dest,
                 default=None,
                 type=None,
                 choices=None,
                 help=None,
                 metavar=None):
        super().__init__(option_strings=option_strings,
                         dest=dest,
                         default=default,
                         type=type,
                         choices=choices,
                         required=True,
                         help=help,
                         metavar=metavar)


def run_subparsers(parser: argparse.ArgumentParser):
    args = parser.parse_args()
    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)
