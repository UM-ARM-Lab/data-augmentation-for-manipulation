#!/usr/bin/env python
import argparse
import csv
import pathlib

import colorama
import numpy as np
import tabulate

import rospy
from link_bot_planning.analysis import results_utils
from link_bot_planning.analysis.results_metrics import any_solved
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_json_or_hjson


def metrics_main(args):
    metadata = load_json_or_hjson(args.results_dir, 'metadata')
    scenario = get_scenario(metadata['scenario'])

    rows = []
    for trial_idx, datum in results_utils.trials_generator(args.results_dir):
        status = datum['trial_status']
        end_state = datum['end_state']
        goal = datum['goal']
        task_error = scenario.distance_to_goal(end_state, goal).numpy()
        used_recovery = False
        recovery_successful = False
        num_recoveries = 0
        steps = datum['steps']
        for step in steps:
            if step['type'] == 'executed_recovery':
                num_recoveries += 1
                used_recovery = True
            if used_recovery and step['type'] == 'executed_plan':
                recovery_successful = True

        solved = any_solved(scenario, metadata, datum)
        row = [trial_idx,
               status.name,
               f'{task_error:.3f}',
               int(used_recovery),
               int(recovery_successful),
               int(solved),
               num_recoveries]
        rows.append(row)

    rows = sorted(rows)

    headers = ['trial idx', 'success?', 'final error', 'used recovery?', 'recovery succeeded?', 'solved?',
               '# recoveries']
    print(tabulate.tabulate(rows, headers=headers, tablefmt=tabulate.simple_separated_format("\t")))

    with open(args.results_dir / 'results_stats.txt', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)
        writer.writerows(rows)


def main():
    colorama.init(autoreset=True)

    rospy.init_node("print_results_stats")
    np.set_printoptions(suppress=True, precision=4, linewidth=180)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('results_dir', help='results directory', type=pathlib.Path)

    args = parser.parse_args()

    metrics_main(args)


if __name__ == '__main__':
    main()
