#!/usr/bin/env python
import argparse
import pathlib

import colorama

from link_bot_planning.test_scenes import TestScene, get_all_scenes
from link_bot_pycommon.args import int_set_arg


def remove_main(args):
    for idx in args.idx:
        try:
            s_i = TestScene(root=args.dirname, idx=idx)
            s_i.remove()
        except FileNotFoundError:
            print(f"could not remove scene {idx}")
            pass


def print_main(args):
    scenes = get_all_scenes(args.dirname)
    for s in scenes:
        print(s.idx)
        print(s.links_states)
        print(s.joint_state)


def list_main(args):
    scenes = get_all_scenes(args.dirname)
    for s in scenes:
        print(s.idx)


def consolidate_main(args):
    scenes = get_all_scenes(args.dirname)
    for new_idx, s in enumerate(scenes):
        s.delete()
        s.change_index(new_idx, force=True)


def double_main(args):
    n_existing_scenes = len(list(args.dirname.glob("*.bag")))

    for i in range(n_existing_scenes):
        j = i + n_existing_scenes
        s_i = TestScene(root=args.dirname, idx=i)
        s_i.change_index(j, force=False)


def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    double_parser = subparsers.add_parser('double')
    double_parser.add_argument("dirname", type=pathlib.Path)
    double_parser.set_defaults(func=double_main)

    remove_parser = subparsers.add_parser('remove')
    remove_parser.add_argument("dirname", type=pathlib.Path)
    remove_parser.add_argument("idx", type=int_set_arg)
    remove_parser.set_defaults(func=remove_main)

    list_parser = subparsers.add_parser('list')
    list_parser.add_argument("dirname", type=pathlib.Path)
    list_parser.set_defaults(func=list_main)

    consolidate_parser = subparsers.add_parser('consolidate')
    consolidate_parser.add_argument("dirname", type=pathlib.Path)
    consolidate_parser.set_defaults(func=consolidate_main)

    print_parser = subparsers.add_parser('print')
    print_parser.add_argument("dirname", type=pathlib.Path)
    print_parser.add_argument("idx", type=int_set_arg)
    print_parser.set_defaults(func=print_main)

    args = parser.parse_args()

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
