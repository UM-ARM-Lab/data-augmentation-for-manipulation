import argparse
import pathlib
import pickle
from time import time

from tqdm import tqdm

from arc_utilities import ros_init
from link_bot_data.dataset_utils import replaced_true_with_predicted
from link_bot_data.load_dataset import get_classifier_dataset_loader
from merrrt_visualization.rviz_animation_controller import RvizAnimationController


@ros_init.with_ros("viz_nearest_examples")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_dataset_dirs", type=pathlib.Path)
    parser.add_argument("reference_example_idx", type=int)
    parser.add_argument("query_dataset_dirs", type=pathlib.Path, nargs='+')
    parser.add_argument("--n-nearest", '-n', type=int, default=100)
    parser.add_argument("--dist-threshold", '-d', type=float, default=0.1)

    args = parser.parse_args()

    ref_loader = get_classifier_dataset_loader([args.reference_dataset_dirs])
    ref_example = ref_loader.get_datasets('all').get_example(args.reference_example_idx)

    s = ref_loader.get_scenario()

    query_loader = get_classifier_dataset_loader(args.query_dataset_dirs)
    query_dataset = query_loader.get_datasets('all')

    nearby_examples = []
    for q_example in tqdm(query_dataset):
        ref_s = replaced_true_with_predicted(ref_example)
        q_s = replaced_true_with_predicted(q_example)
        d = s.classifier_distance(ref_s, q_s)[1]
        if d < args.dist_threshold:
            nearby_examples.append(q_example)
        if len(nearby_examples) > args.n_nearest:
            continue

    outfile = pathlib.Path("results") / f"viz_nearest_examples_{args.reference_example_idx}_{int(time())}"
    print(f"Saving to {outfile.as_posix()}")
    with outfile.open("wb") as f:
        pickle.dump(nearby_examples, f)

    anim = RvizAnimationController()
    while not anim.done:
        i = anim.t()

        anim.step()


if __name__ == '__main__':
    main()
