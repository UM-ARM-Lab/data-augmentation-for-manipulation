import pathlib
from typing import List

import link_bot_classifiers
from link_bot_classifiers.train_test_classifier import setup_datasets
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_pycommon.pycommon import paths_to_json
from moonshine.filepath_tools import load_trial, create_trial
from moonshine.model_runner import ModelRunner


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
                         trials_directory: pathlib.Path = pathlib.Path("./trials"),
                         **kwargs):
    _, model_hparams = load_trial(trial_path=checkpoint.parent.absolute())
    model_hparams['datasets'].extend(paths_to_json(dataset_dirs))

    trial_path, _ = create_trial(log, model_hparams, trials_directory=trials_directory)

    model_class = link_bot_classifiers.get_model(model_hparams['model_class'])

    train_dataset = ClassifierDatasetLoader(dataset_dirs, use_gt_rope=True, load_true_states=True)
    val_dataset = ClassifierDatasetLoader(dataset_dirs, use_gt_rope=True, load_true_states=True)

    # decrease the learning rate, this is often done in fine-tuning
    model_hparams['learning_rate'] = 1e-4  # normally 1e-3
    model = model_class(hparams=model_hparams, batch_size=batch_size, scenario=train_dataset.scenario)
    runner = ModelRunner(model=model,
                         training=True,
                         params=model_hparams,
                         checkpoint=checkpoint,
                         batch_metadata=train_dataset.batch_metadata,
                         early_stopping=early_stopping,
                         trial_path=trial_path,
                         **kwargs)

    train_tf_dataset, val_tf_dataset = setup_datasets(model_hparams, batch_size, train_dataset, val_dataset)

    # Modify the model for feature transfer & fine-tuning
    for c in model.conv_layers:
        c.trainable = fine_tune_conv
    for d in model.dense_layers:
        d.trainable = fine_tune_dense
    model.lstm.trainable = fine_tune_lstm
    model.output_layer.trainable = fine_tune_output

    runner.reset_best_ket_metric_value()
    runner.train(train_tf_dataset, val_tf_dataset, num_epochs=epochs)

    return trial_path
