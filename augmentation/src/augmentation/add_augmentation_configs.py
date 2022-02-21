import itertools
import pathlib
import pickle
from typing import Dict

from link_bot_data.dataset_utils import add_new
from moonshine.torch_and_tf_utils import repeat


def make_add_augmentation_env_func(augmentation_config_dir, batch_size):
    augmentation_config_gen = load_augmentation_configs(augmentation_config_dir)

    def _add_augmentation_env(example: Dict):
        if augmentation_config_dir is not None:
            config = next(augmentation_config_gen)
            # add batching
            new_example = config['env']
            new_example = repeat(new_example, batch_size, axis=0, new_axis=True)
        else:
            new_example = example.copy()

        example[add_new('env')] = new_example['env']
        example[add_new('extent')] = new_example['extent']
        example[add_new('res')] = new_example['res']
        example[add_new('origin')] = new_example['origin']
        example[add_new('origin_point')] = new_example['origin_point']
        example[add_new('scene_msg')] = new_example['scene_msg']
        if 'sdf' in new_example:
            example[add_new('sdf')] = new_example['sdf']
        if 'sdf_grad' in new_example:
            example[add_new('sdf_grad')] = new_example['sdf_grad']

        return example

    return _add_augmentation_env


def add_augmentation_configs_to_dataset(augmentation_config_dir, dataset, batch_size):
    _add_augmentation_env = make_add_augmentation_env_func(augmentation_config_dir, batch_size)
    return dataset.map(_add_augmentation_env)


def load_augmentation_configs(augmentation_config_dir: pathlib.Path):
    augmentation_configs = []
    if augmentation_config_dir is not None:
        # load the pkl files and add them to the dataset?
        augmentation_config_dir = pathlib.Path(augmentation_config_dir)
        for filename in augmentation_config_dir.glob("initial_config*.pkl"):
            with filename.open("rb") as file:
                augmentation_config = pickle.load(file)
                augmentation_configs.append(augmentation_config)
    augmentation_config_gen = itertools.cycle(augmentation_configs)
    return augmentation_config_gen
