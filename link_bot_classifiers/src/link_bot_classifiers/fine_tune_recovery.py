import pathlib
from typing import List, Optional, Dict

import link_bot_classifiers
import link_bot_classifiers.get_model
from arc_utilities.algorithms import nested_dict_update
from link_bot_classifiers.train_test_recovery import setup_datasets
from link_bot_data.recovery_dataset import RecoveryDatasetLoader
from link_bot_pycommon.pycommon import paths_to_json
from moonshine.filepath_tools import load_trial, create_trial
from moonshine.model_runner import ModelRunner


def fine_tune_recovery(dataset_dirs: List[pathlib.Path],
                       checkpoint: pathlib.Path,
                       log: str,
                       batch_size: int,
                       epochs: int,
                       early_stopping: bool,
                       fine_tune_conv: bool,
                       fine_tune_dense: bool,
                       fine_tune_output1: bool,
                       fine_tune_output2: bool,
                       model_hparams_update: Optional[Dict] = None,
                       trials_directory: pathlib.Path = pathlib.Path("./trials"),
                       take: Optional[int] = None,
                       **kwargs):
    _, model_hparams = load_trial(trial_path=checkpoint.parent.absolute())
    model_hparams['datasets'].extend(paths_to_json(dataset_dirs))
    model_hparams = nested_dict_update(model_hparams, model_hparams_update)

    trial_path, _ = create_trial(log, model_hparams, trials_directory=trials_directory)

    model_class = link_bot_classifiers.get_model.get_model(model_hparams['model_class'])

    train_dataset = RecoveryDatasetLoader(dataset_dirs)
    val_dataset = RecoveryDatasetLoader(dataset_dirs)

    # decrease the learning rate, this is often done in fine-tuning
    model_hparams['learning_rate'] = 1e-4  # normally 1e-3
    model = model_class(hparams=model_hparams, batch_size=batch_size, scenario=train_dataset.scenario)

    # override arbitrary parts of the model
    for k, v in kwargs.items():
        if v is not None:
            if hasattr(model, k):
                setattr(model, k, v)

    runner = ModelRunner(model=model,
                         training=True,
                         params=model_hparams,
                         checkpoint=checkpoint,
                         batch_metadata=train_dataset.batch_metadata,
                         early_stopping=early_stopping,
                         trial_path=trial_path,
                         **kwargs)

    train_tf_dataset, val_tf_dataset = setup_datasets(model_hparams, batch_size, train_dataset, val_dataset, take)

    for c in model.conv_layers:
        c.trainable = fine_tune_conv
    for d in model.dense_layers:
        d.trainable = fine_tune_dense
    model.output_layer1.trainable = fine_tune_output1
    model.output_layer2.trainable = fine_tune_output2

    runner.reset_best_ket_metric_value()
    runner.train(train_tf_dataset, val_tf_dataset, num_epochs=epochs)

    return trial_path
