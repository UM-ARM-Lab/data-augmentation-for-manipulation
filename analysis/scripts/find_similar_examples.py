import argparse
import pathlib
import pickle

import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from arc_utilities import ros_init
from link_bot_data.new_dataset_utils import check_download
from link_bot_pycommon.matplotlib_utils import adjust_lightness
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.numpify import numpify
from moonshine.torch_and_tf_utils import remove_batch
from moonshine.torch_datasets_utils import my_collate
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset, remove_keys
from state_space_dynamics.train_test_dynamics import load_udnn_model_wrapper


@ros_init.with_ros("find_similar_examples")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_dataset_dir', type=pathlib.Path)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('train_dataset_dir', type=pathlib.Path)
    parser.add_argument('--load', action='store_true')

    args = parser.parse_args()

    transform = transforms.Compose([remove_keys("scene_msg")])
    eval_dataset = TorchDynamicsDataset(check_download(args.eval_dataset_dir), mode='test', transform=transform)
    train_dataset = TorchDynamicsDataset(check_download(args.train_dataset_dir), mode='train', transform=transform)

    eval_loader = DataLoader(eval_dataset, collate_fn=my_collate)
    train_loader = DataLoader(train_dataset, collate_fn=my_collate)

    model = load_udnn_model_wrapper(args.checkpoint)
    model.eval()

    s = model.scenario

    if args.load:
        with open("similar_examples.pkl", 'rb') as f:
            things_to_viz = pickle.load(f)
    else:
        things_to_viz = generate_data(eval_dataset, eval_loader, model, s, train_dataset, train_loader)
        with open("similar_examples.pkl", 'wb') as f:
            pickle.dump(things_to_viz, f)

    anim = RvizAnimationController(n_time_steps=len(things_to_viz), ns='trajs')

    def _l(c):
        return adjust_lightness(c, 0.5)

    while not anim.done:
        data = things_to_viz[anim.t()]
        s.plot_state_rviz(data['eval']['before'], label='eval_before', color='#ff0000')
        s.plot_state_rviz(data['eval']['after'], label='eval_after', color=_l('#ff0000'))
        s.plot_state_rviz(data['nearest_train']['before'], label='nearest_train_before', color='#ff00ff')
        s.plot_state_rviz(data['nearest_train']['after'], label='nearest_train_after', color=_l('#ff00ff'))
        anim.step()


def generate_data(eval_dataset, eval_loader, model, s, train_dataset, train_loader):
    things_to_viz = []
    # Look at the examples from known_good_2
    for eval_example in tqdm(eval_loader):
        eval_example_local = s.put_state_local_frame_torch(eval_example)
        outputs = model(eval_example)
        error_batch = model.scenario.classifier_distance_torch(eval_example, outputs)
        # with the highest error for the best model (random_planning_close_0.2_low_initial_error-cg889)
        for t in range(1, 10):
            if error_batch[0, t] > 0.1:
                # consider the transition from t-1 to t
                lowest_distance = np.inf
                nearest_example = None
                for train_example in train_loader:
                    train_example_local = s.put_state_local_frame_torch(train_example)
                    train_rope_local_before = train_example_local['rope'][0, t - 1]
                    eval_rope_local_before = eval_example_local['rope'][0, t - 1]
                    distance_before = np.linalg.norm(train_rope_local_before - eval_rope_local_before)
                    train_rope_local_after = train_example_local['rope'][0, t]
                    eval_rope_local_after = eval_example_local['rope'][0, t]
                    distance_after = np.linalg.norm(train_rope_local_after - eval_rope_local_after)
                    distance = distance_before + distance_after
                    if distance < lowest_distance:
                        lowest_distance = distance
                        nearest_example = train_example

                eval_example_viz = numpify(remove_batch(eval_example))
                nearest_example_viz = numpify(remove_batch(nearest_example))
                print(f"Nearest example d={lowest_distance}")
                eval_actual_before = eval_dataset.index_time(eval_example_viz, t - 1)
                eval_actual_after = eval_dataset.index_time(eval_example_viz, t)
                train_actual_before = train_dataset.index_time(nearest_example_viz, t - 1)
                train_actual_after = train_dataset.index_time(nearest_example_viz, t)
                things_to_viz.append({
                    'eval':          {
                        'before': eval_actual_before,
                        'after':  eval_actual_after,
                    },
                    'nearest_train': {
                        'before': train_actual_before,
                        'after':  train_actual_after,
                    }
                })
    return things_to_viz


if __name__ == '__main__':
    main()
