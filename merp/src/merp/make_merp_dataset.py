import pathlib
from typing import Optional

from merp.torch_merp_dataset import TorchMERPDataset

from link_bot_data.classifier_dataset_utils import PredictionActualExample, int_scalar_to_batched_float
from link_bot_data.dataset_utils import add_predicted
from link_bot_data.tf_dataset_utils import write_example, deserialize_scene_msg
from link_bot_pycommon.load_wandb_model import load_model_artifact
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from link_bot_pycommon.serialization import my_hdump
from moonshine.filepath_tools import load_params
from state_space_dynamics.udnn_torch import UDNN


def make_merp_dataset(dataset_dir: pathlib.Path,
                      checkpoint: pathlib.Path,
                      outdir: pathlib.Path,
                      batch_size: Optional[int] = None):
    model = load_model_artifact(checkpoint, UDNN, project='udnn', version='latest', user='armlab')

    merp_dataset_hparams = load_params(dataset_dir)

    merp_dataset_hparams['dataset_dir'] = dataset_dir.as_posix()
    merp_dataset_hparams['fwd_model_hparams'] = model.hparams
    merp_dataset_hparams['predicted_state_keys'] = model.state_keys

    new_hparams_filename = outdir / 'hparams.hjson'
    my_hdump(merp_dataset_hparams, new_hparams_filename.open("w"), indent=2)

    total_example_idx = 0
    for mode in ['train', 'val', 'test']:
        dataset = TorchMERPDataset(dataset_dir, mode=mode)

        for out_examples in generate_classifier_examples(model, dataset, batch_size):
            for out_examples_start_t in out_examples:
                actual_batch_size = out_examples_start_t['traj_idx'].shape[0]
                for batch_idx in range(actual_batch_size):
                    out_example_b = index_dict_of_batched_tensors(out_examples_start_t, batch_idx)
                    out_example_b['metadata'] = {k: v[batch_idx] for k, v in out_examples_start_t['metadata'].items()}

                    if out_example_b['time_idx'].ndim == 0:
                        continue

                    write_example(outdir, out_example_b, total_example_idx, save_format='pkl')
                    total_example_idx += 1

    return outdir


def generate_classifier_examples(model,
                                 dataset,
                                 batch_size: int):
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)

    for idx, example in enumerate(dataset):
        deserialize_scene_msg(example)

        valid_out_examples = []
        for start_t in range(0, dataset_loader.steps_per_traj - classifier_horizon + 1, labeling_params['start_step']):
            prediction_end_t = dataset_loader.steps_per_traj
            actual_prediction_horizon = prediction_end_t - start_t
            dataset_loader.state_metadata_keys = ['joint_names']  # NOTE: perhaps ACOs should be state metadata?
            state_keys = dataset_loader.state_keys + dataset_loader.state_metadata_keys
            actual_states_from_start_t = {k: example[k][:, start_t:prediction_end_t] for k in state_keys}
            actual_start_states = {k: example[k][:, start_t] for k in state_keys}
            actions_from_start_t = {k: example[k][:, start_t:prediction_end_t - 1] for k in dataset_loader.action_keys}
            environment = {k: example[k] for k in dataset_loader.env_keys}

            predictions = model(inputs)

            prediction_actual = PredictionActualExample(example=example,
                                                        actions=actions_from_start_t,
                                                        actual_states=actual_states_from_start_t,
                                                        predictions=predictions_from_start_t,
                                                        start_t=start_t,
                                                        env_keys=dataset_loader.env_keys,
                                                        labeling_params={},
                                                        actual_prediction_horizon=actual_prediction_horizon,
                                                        batch_size=example['batch_size'])
            valid_out_examples_for_start_t = generate_classifier_examples_from_batch(sc, prediction_actual)
            valid_out_examples.extend(valid_out_examples_for_start_t)

        yield valid_out_examples


def generate_classifier_examples_from_batch(scenario: ScenarioWithVisualization,
                                            prediction_actual: PredictionActualExample):
    classifier_horizon = 2
    prediction_horizon = prediction_actual.actual_prediction_horizon
    batch_size = prediction_actual.batch_size

    valid_out_example_batches = []
    for classifier_start_t in range(0, prediction_horizon - classifier_horizon + 1):
        classifier_end_t = classifier_start_t + classifier_horizon

        prediction_start_t = prediction_actual.start_t
        prediction_start_t_batched = int_scalar_to_batched_float(batch_size, prediction_start_t)
        classifier_start_t_batched = int_scalar_to_batched_float(batch_size, classifier_start_t)
        classifier_end_t_batched = int_scalar_to_batched_float(batch_size, classifier_end_t)
        out_example = {
            'traj_idx':           prediction_actual.example['traj_idx'],
            'prediction_start_t': prediction_start_t_batched,
            'classifier_start_t': classifier_start_t_batched,
            'classifier_end_t':   classifier_end_t_batched,
        }
        # add actual to out example
        out_example.update({k: prediction_actual.example[k] for k in prediction_actual.env_keys})

        # add predictions to out example
        for key, prediction_component in prediction_actual.predictions.items():
            out_example[add_predicted(key)] = prediction_component

        # add error
        error = scenario.classifier_distance(prediction_actual.actual_states, prediction_actual.predictions)
        out_example['metadata'] = {
            'error': error['error'],
        }
        valid_out_example_batches.append(out_example)

    return valid_out_example_batches


def split_actual_predicted(out_example, prediction_actual: PredictionActualExample):
    actual = {}
    for key, actual_state_component in prediction_actual.actual_states.items():
        out_example[key] = actual_state_component
        actual[key] = actual_state_component
    sliced_predictions = {}
    for key, action_component in prediction_actual.actions.items():
        out_example[key] = action_component
        actual[key] = action_component
    return actual, sliced_predictions
