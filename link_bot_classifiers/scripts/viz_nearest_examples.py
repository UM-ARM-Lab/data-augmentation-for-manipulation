#!/usr/bin/env python

import argparse
import pathlib
import pickle
from time import time
from typing import Dict

from arc_utilities import ros_init
from link_bot_data.dataset_utils import replaced_true_with_predicted
from link_bot_data.load_dataset import get_classifier_dataset_loader
from merrrt_visualization.rviz_animation_controller import RvizAnimationController


def _keep_nearby(q_example: Dict, *args):
    ref_example, dist_threshold, classifier_distance = args
    ref_s = replaced_true_with_predicted(ref_example)
    q_s = replaced_true_with_predicted(q_example)
    d = classifier_distance(None, ref_s, q_s)[1]
    keep = d < dist_threshold
    return keep


@ros_init.with_ros("viz_nearest_examples")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_dataset_dirs", type=pathlib.Path)
    parser.add_argument("reference_example_idx", type=int)
    parser.add_argument("query_dataset_dirs", type=pathlib.Path, nargs='+')
    parser.add_argument("--n-nearest", '-n', type=int, default=100)
    parser.add_argument("--dist-threshold", '-d', type=float, default=0.3)
    parser.add_argument("--restore", type=pathlib.Path)

    args = parser.parse_args()

    ref_loader = get_classifier_dataset_loader([args.reference_dataset_dirs])
    ref_example = ref_loader.get_datasets('all').get_example(args.reference_example_idx)

    s = ref_loader.get_scenario()

    query_loader = get_classifier_dataset_loader(args.query_dataset_dirs)
    query_dataset = query_loader.get_datasets('all')

    if args.restore:
        with args.restore.open("rb") as f:
            nearby_examples = pickle.load(f)
    else:
        cd_func = s.__class__.classifier_distance
        nearby_examples = list(query_dataset.parallel_filter(_keep_nearby, ref_example, args.dist_threshold, cd_func))

        outfile = pathlib.Path("results") / f"viz_nearest_examples_{args.reference_example_idx}_{int(time())}"
        print(f"Saving to {outfile.as_posix()}")
        with outfile.open("wb") as f:
            pickle.dump(nearby_examples, f)

    anim = RvizAnimationController(n_time_steps=len(nearby_examples))
    while not anim.done:
        i = anim.t()

        nearby_example_i = nearby_examples[i]
        ref_example_s = replaced_true_with_predicted(ref_example)
        nearby_example_s = replaced_true_with_predicted(nearby_example_i)
        d = s.classifier_distance(ref_example_s, nearby_example_s)[1]
        ref_example['error'] = [d, d]
        nearby_example_i['error'] = [d, d]
        ref_loader.plot_transition(ref_example, label='ref', color='black')
        ref_loader.plot_transition(nearby_example_i, label='near', color='blue')

        anim.step()


if __name__ == '__main__':
    main()
