#!/usr/bin/env python
import argparse
import pathlib

import colorama

from arc_utilities import ros_init
from link_bot_gazebo import gazebo_services
from link_bot_planning.test_scenes import TestScene, get_all_scenes
from link_bot_pycommon.args import int_set_arg
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_hjson


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
        if s.idx == args.idx:
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


def viz_main(args):
    scenario = get_scenario(args.scenario)
    scenario.on_before_get_state_or_execute_action()
    scenes = get_all_scenes(args.dirname)
    service_provider = gazebo_services.GazeboServices()
    params = load_hjson(args.params)
    print("Press enter to get the next scene, q to quit")
    service_provider.setup_env()
    for new_idx, s in enumerate(scenes):
        scene_filename = s.get_scene_filename()
        scenario.restore_from_bag(service_provider, params, scene_filename)
        k = input()
        if k == 'q':
            break


def cp_main(args):
    start_idx = len(get_all_scenes(args.dest))
    out_idx = start_idx
    for idx in args.idx:
        # get the source scene
        s_i = TestScene(root=args.src, idx=idx)

        # change its info and save
        s_i.root = args.dest
        s_i.change_index(out_idx)

        out_idx += 1


def filter_main(args):
    scenario = get_scenario(args.scenario)
    scenario.on_before_get_state_or_execute_action()
    scenes = get_all_scenes(args.dirname)
    service_provider = gazebo_services.GazeboServices()
    params = load_hjson(args.params)
    service_provider.setup_env()
    args.outdir.mkdir(exist_ok=True)
    out_idx = 0
    for new_idx, s in enumerate(scenes):
        scene_filename = s.get_scene_filename()
        scenario.restore_from_bag(service_provider, params, scene_filename)
        k = input("Keep? [Y/n]")
        if k in ['n', 'N']:
            continue

        s.root = args.outdir
        s.idx = out_idx
        out_idx += 1
        s.save()


def double_main(args):
    n_existing_scenes = len(list(args.dirname.glob("*.bag")))

    for i in range(n_existing_scenes):
        j = i + n_existing_scenes
        s_i = TestScene(root=args.dirname, idx=i)
        s_i.change_index(j, force=False)


@ros_init.with_ros("scenes")
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
    print_parser.add_argument("idx", type=int)
    print_parser.set_defaults(func=print_main)

    viz_parser = subparsers.add_parser('viz')
    viz_parser.add_argument("dirname", type=pathlib.Path)
    viz_parser.add_argument("params", type=pathlib.Path, help='you can use a planner params hjson for this')
    viz_parser.add_argument("--scenario", default='dual_arm_rope_sim_val_with_robot_feasibility_checking')
    viz_parser.set_defaults(func=viz_main)

    filter_parser = subparsers.add_parser('filter')
    filter_parser.add_argument("dirname", type=pathlib.Path)
    filter_parser.add_argument("params", type=pathlib.Path, help='you can use a planner params hjson for this')
    filter_parser.add_argument("outdir", type=pathlib.Path, help='dir to save the filtered scenes to')
    filter_parser.add_argument("--scenario", default='dual_arm_rope_sim_val_with_robot_feasibility_checking')
    filter_parser.set_defaults(func=filter_main)

    copy_parser = subparsers.add_parser('cp')
    copy_parser.add_argument("src", type=pathlib.Path)
    copy_parser.add_argument("dest", type=pathlib.Path)
    copy_parser.add_argument("idx", type=int_set_arg)
    copy_parser.set_defaults(func=cp_main)

    args = parser.parse_args()

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
