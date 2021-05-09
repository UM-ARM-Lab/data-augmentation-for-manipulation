#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import colorama
import hjson
import matplotlib.pyplot as plt
import numpy as np

from arc_utilities import ros_init
from link_bot_classifiers.test_classifier import test_classifier
from link_bot_pycommon.experiment_scenario import ExperimentScenario


@ros_init.with_ros("test_classifier_given_actions")
def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("fwd_model_dir", help="fwd model dirs", type=pathlib.Path, nargs="+")
    parser.add_argument("classifier_model_dir", help="classifier model dir", type=pathlib.Path)
    parser.add_argument("actions", help="actions hjson file", type=pathlib.Path)
    parser.add_argument("--saved-state", help="bagfile describing the saved state", type=pathlib.Path)

    args = parser.parse_args()

    with args.actions.open("r") as actions_file:
        actions = hjson.load(actions_file)
    n_actions = len(actions)

    def _generate_actions(environment: Dict,
                          start_state_tiled: Dict,
                          scenario: ExperimentScenario,
                          params: Dict,
                          n_actions: int):
        return actions

    test_classifier(classifier_model_dir=args.classifier_model_dir,
                    fwd_model_dir=args.fwd_model_dir,
                    saved_state=args.saved_state,
                    generate_actions=_generate_actions,
                    n_actions=n_actions)


if __name__ == '__main__':
    main()
