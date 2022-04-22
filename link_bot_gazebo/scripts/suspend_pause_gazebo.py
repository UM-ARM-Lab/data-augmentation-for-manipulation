#!/usr/bin/env python
import argparse

from link_bot_gazebo.gazebo_utils import get_gazebo_processes


def suspend(args):
    gazebo_processes = get_gazebo_processes()
    [p.suspend() for p in gazebo_processes]


def resume(args):
    gazebo_processes = get_gazebo_processes()
    [p.resume() for p in gazebo_processes]


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    suspend_parser = subparsers.add_parser('suspend')
    suspend_parser.set_defaults(func=suspend)
    resume_parser = subparsers.add_parser('resume')
    resume_parser.set_defaults(func=resume)


if __name__ == '__main__':
    main()
