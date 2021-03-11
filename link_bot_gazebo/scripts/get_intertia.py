#!/usr/bin/env python
import argparse

import colorama

from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.inertia_matrices import sphere, cylinder, box


def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    subparsers = parser.add_subparsers()
    box_parser = subparsers.add_parser('box', help='box')
    box_parser.add_argument('mass', type=float, help='mass')
    box_parser.add_argument('x', type=float, help='x')
    box_parser.add_argument('y', type=float, help='y')
    box_parser.add_argument('z', type=float, help='z')
    box_parser.set_defaults(func=box)

    sphere_parser = subparsers.add_parser('sphere', help='sphere')
    sphere_parser.add_argument('mass', type=float, help='mass')
    sphere_parser.add_argument('radius', type=float, help='radius')
    sphere_parser.set_defaults(func=sphere)

    cylinder_parser = subparsers.add_parser('cylinder', help='cylinder')
    cylinder_parser.add_argument('mass', type=float, help='mass')
    cylinder_parser.add_argument('radius', type=float, help='radius')
    cylinder_parser.add_argument('length', type=float, help='length')
    cylinder_parser.set_defaults(func=cylinder)

    args = parser.parse_args()
    inertia = args.func(args)
    print(f"<ixx>{inertia[0]:.8f}</ixx>")
    print(f"<iyy>{inertia[1]:.8f}</iyy>")
    print(f"<izz>{inertia[2]:.8f}</izz>")


if __name__ == '__main__':
    main()
