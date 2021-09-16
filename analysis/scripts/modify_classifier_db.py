#!/usr/bin/env python
import argparse
import pathlib

import boto3

from analysis.results_utils import try_load_classifier_params
from link_bot_data import dynamodb_utils
from link_bot_data.dynamodb_utils import update_classifier_db, remove_duplicates_in_classifier_db
from link_bot_pycommon.pycommon import has_keys


def add_fine_tuning_dataset_dirs(item):
    k = 'fine_tuning_dataset_dirs'
    if k in item:
        return None

    classifier_model_dir = pathlib.Path(item['classifier']["S"])
    classifier_hparams = try_load_classifier_params(classifier_model_dir)
    datasets = classifier_hparams['datasets']
    if len(datasets) == 1:
        return True, 'NULL', k
    else:
        default = classifier_hparams['datasets'][-1]
        fine_tuning_dataset_dirs = classifier_hparams.get('fine_tuning_dataset_dirs', default)
        return str(fine_tuning_dataset_dirs), 'S', k


def get_classifier_source_env(item):
    if 'classifier_source_env' in item and "S" in item['classifier_source_env']:
        return None
    try:
        classifier_model_dir = pathlib.Path(item['classifier']["S"])
        classifier_hparams = try_load_classifier_params(classifier_model_dir)
        classifier_source_env = classifier_hparams['classifier_dataset_hparams']['scene_name']
    except Exception:
        classifier_source_env = 'source-env-unknown'
    return classifier_source_env, "S", "classifier_source_env"


def add_fine_tuning_take(item):
    if 'take' in item['classifier']['S']:
        return str(10), 'N', 'fine_tuning_take'
    else:
        return True, 'NULL', 'fine_tuning_take'


def add_lr(item):
    k = 'learning_rate'
    try:
        classifier_model_dir = pathlib.Path(item['classifier']["S"])
        classifier_hparams = try_load_classifier_params(classifier_model_dir)
        v = classifier_hparams.get(k, None)
        if v is None:
            return True, "NULL", k
        return str(v), 'N', 'learning_rate'
    except Exception:
        return True, "NULL", k


def add_time(s):
    return '-1', 'N', 'time'


def add_fine_tuning_layers(s):
    def _f(item):
        k = f'fine_tune_{s}'

        try:
            classifier_model_dir = pathlib.Path(item['classifier']["S"])
            classifier_hparams = try_load_classifier_params(classifier_model_dir)
            v = classifier_hparams.get(k, None)
            if v is None:
                return True, "NULL", k
            return v, "BOOL", k
        except Exception:
            return True, "NULL", k

    return _f


def update_do_augmentation(item):
    if 'do_augmentation' not in item or list(item['do_augmentation'].keys())[0] == 'NULL':
        return str(0), 'N', 'do_augmentation'
    return None


def update_original_seed(item):
    try:
        if 'original_trining_seed' not in item:
            classifier_model_dir = pathlib.Path(item['classifier']["S"])
            classifier_hparams = try_load_classifier_params(classifier_model_dir)
            seed = classifier_hparams['seed']
            return str(seed), "N", "original_training_seed"
        else:
            return None
    except Exception:
        return True, "NULL", "original_training_seed"


def update_fine_tuning_seed(item):
    if 'fine_tuning_seed' not in item or 'NULL' in item['fine_tuning_seed']:
        print(item['classifier']['S'])
        q = input("Aug?")
        if q in ['Y', 'y', '1']:
            return str(0), "N", "fine_tuning_seed"
        else:
            return True, "NULL", "fine_tuning_seed"
    else:
        return None


def rename_classifier_model_dir(item):
    classifier_model_dir = pathlib.Path(item['classifier']["S"])
    potential_classifier_model_dirs = [
        classifier_model_dir,
        pathlib.Path('/media/shared/cl_trials') / pathlib.Path(*classifier_model_dir.parent.parts[1:]),
        pathlib.Path('/media/shared/ift') / pathlib.Path(*classifier_model_dir.parent.parts[2:])
    ]
    classifier_model_dir_out = classifier_model_dir
    for d in potential_classifier_model_dirs:
        if d.exists():
            classifier_model_dir_out = d
            break
    if 'best_checkpoint' not in classifier_model_dir_out.as_posix():
        classifier_model_dir_out = classifier_model_dir_out / 'best_checkpoint'
    return classifier_model_dir_out.as_posix(), "S", "classifier"


def unlistify_fine_tuning_dataset_dirs(item):
    k = 'fine_tuning_dataset_dirs'
    if 'NULL' in item[k]:
        return None
    if 'S' in item[k]:
        return None
    if 'L' in item[k]:
        d = item[k]['L']
        d0 = d[0]['S']
        return str(d0), 'S', k


def add_aug_type(item):
    k = 'augmentation_type'
    if "NULL" in item['fine_tuning_dataset_dirs']:
        return True, 'NULL', k
    else:
        return 'optimization2', 'S', k


def add_invariance_model(item):
    k = 'invariance_model'
    if "N" in item['do_augmentation'] and item['do_augmentation']['N'] == '1':
        return '/media/shared/invariance_trials/relu/July_16_11-30-35_47c46e451e/best_checkpoint', 'S', k
    else:
        return True, 'NULL', k


def add_on_invalid_aug(item):
    k = 'on_invalid_aug'
    if "N" in item['do_augmentation'] and item['do_augmentation']['N'] == '1':
        classifier_model_dir = pathlib.Path(item['classifier']["S"])
        classifier_hparams = try_load_classifier_params(classifier_model_dir)
        v = classifier_hparams['augmentation'].get(k, None)
        if v is None:
            return 'original', "S", k
        return v, "S", k
    else:
        return True, 'NULL', k


def add_ift_uuid(item):
    k = 'ift_uuid'
    if "S" in item['classifier']:
        classifier_model_dir = pathlib.Path(item['classifier']["S"])
        classifier_hparams = try_load_classifier_params(classifier_model_dir)
        if classifier_hparams is None:
            return True, 'NULL', k
        v = classifier_hparams.get(k, None)
        if v is None:
            return True, 'NULL', k
        return v, "S", k
    else:
        return True, 'NULL', k


def add_fine_tuned_from(item):
    k = 'fine_tuned_from'
    if "S" in item['fine_tuning_dataset_dirs']:
        classifier_model_dir = pathlib.Path(item['classifier']["S"])
        classifier_hparams = try_load_classifier_params(classifier_model_dir)
        v = classifier_hparams.get(k, None)
        if v is None:
            return 'unknown', "S", k
        return v, "S", k
    else:
        return True, 'NULL', k


def add_full_retrain(item):
    k = 'full_retrain'
    if "S" in item['classifier']:
        classifier_model_dir = pathlib.Path(item['classifier']["S"])
        classifier_hparams = try_load_classifier_params(classifier_model_dir)
        if classifier_hparams is None:
            return True, 'NULL', k
        v = has_keys(classifier_hparams, ['ift_config', 'full_retrain_classifier'])
        if v is None:
            return True, 'NULL', k
        return v, "BOOL", k
    else:
        return True, 'NULL', k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug')
    args = parser.parse_args()

    table = dynamodb_utils.classifier_table(args.debug)

    client = boto3.client('dynamodb')

    # q = input(Fore.RED + f"MAKES SURE NO OTHER PROCESSES ARE UPDATING TABLE {table}" + Fore.RESET)
    # if q not in ['y', 'Y']:
    #     print("Aborting")
    #     return

    # update_classifier_db(client, table, add_full_retrain)
    remove_duplicates_in_classifier_db(client, table)


if __name__ == '__main__':
    main()
