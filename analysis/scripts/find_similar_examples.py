import argparse
import pathlib
import pickle

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from arc_utilities import ros_init
from link_bot_data.new_dataset_utils import fetch_udnn_dataset
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
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
    eval_dataset = TorchDynamicsDataset(fetch_udnn_dataset(args.eval_dataset_dir), mode='test', transform=transform)
    train_dataset = TorchDynamicsDataset(fetch_udnn_dataset(args.train_dataset_dir), mode='train', transform=transform)

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

    while not anim.done:
        data = things_to_viz.iloc[anim.t()]
        s.plot_state_rviz(data['eval'], label='eval', color='#ff0000')
        s.plot_state_rviz(data['train'], label='nearest_train', color='#ff00ff')
        s.plot_error_rviz(data['state_distance'])
        anim.step()


def generate_data(eval_dataset, eval_loader, model, s, train_dataset, train_loader):
    data = []
    # Look at the examples from known_good_2
    for eval_example in tqdm(eval_loader):
        eval_example_local = s.put_state_local_frame_torch(eval_example)
        outputs = model(eval_example)
        error_batch = model.scenario.classifier_distance_torch(eval_example, outputs)
        # with the highest error for the best model (random_planning_close_0.2_low_initial_error-cg889)
        for t in range(0, 9):
            if error_batch[0, t] > 0.1:
                # consider the transition from t-1 to t
                lowest_distance = np.inf
                lowest_state_distance = None
                nearest_train_example_local = None
                lowest_action_distance = None
                for train_example in train_loader:
                    train_example_local = s.put_state_local_frame_torch(train_example)
                    train_rope_local = train_example_local['rope'][0, t]
                    eval_rope_local = eval_example_local['rope'][0, t]
                    train_left_gripper_delta = train_example['left_gripper_position'][0, t] - \
                                               train_example['left_gripper'][0, t]
                    eval_left_gripper_delta = eval_example['left_gripper_position'][0, t] - \
                                              eval_example['left_gripper'][0, t]
                    train_right_gripper_delta = train_example['right_gripper_position'][0, t] - \
                                                train_example['right_gripper'][0, t]
                    eval_right_gripper_delta = eval_example['right_gripper_position'][0, t] - \
                                               eval_example['right_gripper'][0, t]
                    state_distance = np.linalg.norm(train_rope_local - eval_rope_local)
                    left_gripper_distance = np.linalg.norm(train_left_gripper_delta - eval_left_gripper_delta)
                    right_gripper_distance = np.linalg.norm(train_right_gripper_delta - eval_right_gripper_delta)
                    action_distance = left_gripper_distance + right_gripper_distance
                    distance = state_distance + action_distance
                    if distance < lowest_distance:
                        lowest_distance = distance
                        lowest_state_distance = state_distance
                        lowest_action_distance = action_distance
                        nearest_train_example_local = train_example_local

                eval_example_local_t = eval_dataset.index_time(remove_batch(eval_example_local), t)
                nearest_train_example_local_t = train_dataset.index_time(remove_batch(nearest_train_example_local), t)

                data.append([
                    eval_example_local_t,
                    nearest_train_example_local_t,
                    lowest_state_distance,
                    lowest_action_distance,
                ])
    return pd.DataFrame(data, columns=['eval', 'train', 'state_distance', 'action_distance'])


if __name__ == '__main__':
    main()
