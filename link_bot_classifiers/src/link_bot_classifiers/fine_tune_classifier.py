import itertools
import pathlib
import pickle
from typing import List, Optional, Dict

import link_bot_classifiers
from arc_utilities.algorithms import nested_dict_update
from link_bot_classifiers.train_test_classifier import setup_datasets
from link_bot_data.dataset_utils import add_new
from link_bot_data.load_dataset import load_classifier_dataset
from link_bot_pycommon.pycommon import paths_to_json
from moonshine.filepath_tools import load_trial, create_trial
from moonshine.model_runner import ModelRunner
from moonshine.moonshine_utils import repeat


def fine_tune_classifier(dataset_dirs: List[pathlib.Path],
                         checkpoint: pathlib.Path,
                         log: str,
                         batch_size: int,
                         epochs: int,
                         early_stopping: bool,
                         fine_tune_conv: bool,
                         fine_tune_dense: bool,
                         fine_tune_lstm: bool,
                         fine_tune_output: bool,
                         model_hparams_update: Optional[Dict] = None,
                         verbose: int = 0,
                         trials_directory: pathlib.Path = pathlib.Path("./trials"),
                         augmentation_config_dir: Optional[pathlib.Path] = None,
                         profile: Optional[tuple] = None,
                         take: Optional[int] = None,
                         **kwargs):
    _, model_hparams = load_trial(trial_path=checkpoint.parent.absolute())
    model_hparams['datasets'].extend(paths_to_json(dataset_dirs))
    model_hparams = nested_dict_update(model_hparams, model_hparams_update)

    trial_path, _ = create_trial(log, model_hparams, trials_directory=trials_directory)

    model_class = link_bot_classifiers.get_model(model_hparams['model_class'])

    train_dataset_loader = load_classifier_dataset(dataset_dirs, use_gt_rope=True, load_true_states=True,
                                                   verbose=verbose)
    val_dataset_loader = load_classifier_dataset(dataset_dirs, use_gt_rope=True, load_true_states=True, verbose=verbose)

    # decrease the learning rate, this is often done in fine-tuning
    model_hparams['learning_rate'] = 1e-4  # normally 1e-3
    model = model_class(hparams=model_hparams, batch_size=batch_size, scenario=train_dataset_loader.scenario)

    # override arbitrary parts of the model
    for k, v in kwargs.items():
        if v is not None:
            if hasattr(model, k):
                setattr(model, k, v)

    runner = ModelRunner(model=model,
                         training=True,
                         params=model_hparams,
                         checkpoint=checkpoint,
                         batch_metadata=train_dataset_loader.batch_metadata,
                         early_stopping=early_stopping,
                         profile=profile,
                         trial_path=trial_path,
                         **kwargs)

    train_dataset, val_dataset = setup_datasets(model_hparams,
                                                batch_size=batch_size,
                                                train_dataset_loader=train_dataset_loader,
                                                val_dataset_loader=val_dataset_loader,
                                                seed=0,
                                                take=take)

    if augmentation_config_dir is not None:
        train_dataset = add_augmentation_configs_to_dataset(augmentation_config_dir, train_dataset, batch_size)
        val_dataset = add_augmentation_configs_to_dataset(augmentation_config_dir, val_dataset, batch_size)
    else:
        print("Warning, augmentation_config_dir is None")

    # Modify the model for feature transfer & fine-tuning
    for c in model.conv_layers:
        c.trainable = fine_tune_conv
    for d in model.dense_layers:
        d.trainable = fine_tune_dense
    model.lstm.trainable = fine_tune_lstm
    model.output_layer.trainable = fine_tune_output

    runner.reset_best_ket_metric_value()
    runner.train(train_dataset, val_dataset, num_epochs=epochs)

    return trial_path


def add_augmentation_configs_to_dataset(augmentation_config_dir, dataset, batch_size):
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

        return example

    return dataset.map(_add_augmentation_env)


def load_augmentation_configs(augmentation_config_dir: pathlib.Path):
    augmentation_configs = []
    if augmentation_config_dir is not None:
        # load the pkl files and add them to the dataset?
        augmentation_config_dir = pathlib.Path(augmentation_config_dir)
        for filename in augmentation_config_dir.glob("initial_config*.pkl"):
            with filename.open("rb") as file:
                augmentation_config = pickle.load(file)
                augmentation_config['env'].pop("scene_msg")
                augmentation_configs.append(augmentation_config)
    augmentation_config_gen = itertools.cycle(augmentation_configs)
    return augmentation_config_gen
