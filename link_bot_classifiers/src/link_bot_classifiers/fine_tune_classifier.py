import itertools
import logging
import pathlib
import pickle
from typing import List, Optional, Dict

import link_bot_classifiers
import link_bot_classifiers.get_model
from arc_utilities.algorithms import nested_dict_update
from link_bot_classifiers.train_test_classifier import setup_datasets
from link_bot_data.dataset_utils import add_new
from link_bot_data.load_dataset import get_classifier_dataset_loader
from link_bot_pycommon.pycommon import paths_to_json
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.filepath_tools import load_trial, create_trial
from moonshine.model_runner import ModelRunner
from moonshine.moonshine_utils import repeat

logger = logging.getLogger(__file__)


def fine_tune_classifier(train_dataset_dirs: List[pathlib.Path],
                         checkpoint: pathlib.Path,
                         log: str,
                         batch_size: int,
                         epochs: int,
                         early_stopping: bool,
                         fine_tune_conv: bool,
                         fine_tune_dense: bool,
                         fine_tune_lstm: bool,
                         fine_tune_output: bool,
                         val_dataset_dirs: Optional[List[pathlib.Path]] = None,
                         model_hparams_update: Optional[Dict] = None,
                         verbose: int = 0,
                         trials_directory: pathlib.Path = pathlib.Path("./trials"),
                         augmentation_config_dir: Optional[pathlib.Path] = None,
                         profile: Optional[tuple] = None,
                         take: Optional[int] = None,
                         seed: int = 0,
                         **kwargs):
    train_dataset_loader = get_classifier_dataset_loader(train_dataset_dirs, load_true_states=True, verbose=verbose)
    if val_dataset_dirs is None:
        val_dataset_dirs = train_dataset_dirs
    val_dataset_loader = get_classifier_dataset_loader(val_dataset_dirs, load_true_states=True, verbose=verbose)

    train_dataset = train_dataset_loader.get_datasets(mode='train', shuffle=seed)
    val_dataset = val_dataset_loader.get_datasets(mode='val', shuffle=seed)

    return fine_tune_classifier_from_datasets(train_dataset=train_dataset,
                                              val_dataset=val_dataset,
                                              checkpoint=checkpoint,
                                              log=log,
                                              scenario=train_dataset_loader.get_scenario(),
                                              train_dataset_dirs=train_dataset_dirs,
                                              batch_metadata=train_dataset_loader.batch_metadata,
                                              batch_size=batch_size,
                                              epochs=epochs,
                                              early_stopping=early_stopping,
                                              fine_tune_conv=fine_tune_conv,
                                              fine_tune_dense=fine_tune_dense,
                                              fine_tune_lstm=fine_tune_lstm,
                                              fine_tune_output=fine_tune_output,
                                              model_hparams_update=model_hparams_update,
                                              trials_directory=trials_directory,
                                              augmentation_config_dir=augmentation_config_dir,
                                              profile=profile,
                                              take=take,
                                              seed=seed,
                                              **kwargs)


def fine_tune_classifier_from_datasets(train_dataset,
                                       val_dataset,
                                       checkpoint: pathlib.Path,
                                       log: str,
                                       scenario: ScenarioWithVisualization,
                                       train_dataset_dirs: List[pathlib.Path],
                                       batch_metadata: Dict,
                                       batch_size: int,
                                       epochs: int,
                                       early_stopping: bool = True,
                                       fine_tune_conv: bool = False,
                                       fine_tune_dense: bool = False,
                                       fine_tune_lstm: bool = False,
                                       fine_tune_output: bool = True,
                                       model_hparams_update: Optional[Dict] = None,
                                       trials_directory: pathlib.Path = pathlib.Path("./trials"),
                                       augmentation_config_dir: Optional[pathlib.Path] = None,
                                       profile: Optional[tuple] = None,
                                       take: Optional[int] = None,
                                       seed: Optional[int] = None,
                                       **kwargs):
    _, model_hparams = load_trial(trial_path=checkpoint.parent.absolute())
    model_hparams['datasets'].extend(paths_to_json(train_dataset_dirs))
    model_hparams['fine_tuning_seed'] = seed
    model_hparams['fine_tuning_take'] = take
    model_hparams['fine_tuning_dataset_dirs'] = train_dataset_dirs
    model_hparams = nested_dict_update(model_hparams, model_hparams_update)
    model_class = link_bot_classifiers.get_model.get_model(model_hparams['model_class'])
    # decrease the learning rate, this is often done in fine-tuning
    model_hparams['learning_rate'] = 1e-4  # normally 1e-3
    model = model_class(hparams=model_hparams, batch_size=batch_size, scenario=scenario)
    # override arbitrary parts of the model
    for k, v in kwargs.items():
        if v is not None:
            if hasattr(model, k):
                setattr(model, k, v)
    trial_path, _ = create_trial(log, model_hparams, trials_directory=trials_directory)
    runner = ModelRunner(model=model,
                         training=True,
                         params=model_hparams,
                         checkpoint=checkpoint,
                         batch_metadata=batch_metadata,
                         early_stopping=early_stopping,
                         profile=profile,
                         trial_path=trial_path,
                         **kwargs)
    val_take = kwargs.get('val_take', take)
    train_dataset, val_dataset = setup_datasets(model_hparams=model_hparams,
                                                batch_size=batch_size,
                                                train_dataset=train_dataset,
                                                val_dataset=val_dataset,
                                                seed=seed,
                                                train_take=take,
                                                val_take=val_take)
    if augmentation_config_dir is not None:
        train_dataset = add_augmentation_configs_to_dataset(augmentation_config_dir, train_dataset, batch_size)
    else:
        logger.warning("augmentation_config_dir is None")
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
        example[add_new('scene_msg')] = new_example['scene_msg']

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
                augmentation_configs.append(augmentation_config)
    augmentation_config_gen = itertools.cycle(augmentation_configs)
    return augmentation_config_gen
