#!/usr/bin/env python
import argparse
import pathlib

import boto3

from analysis.results_utils import try_load_classifier_params
from link_bot_data import dynamodb_utils
from link_bot_data.dynamodb_utils import update_classifier_db


def get_classifier_source_env(item):
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


def update_do_augmentation(item):
    if 'do_augmentation' not in item or list(item['do_augmentation'].keys())[0] == 'N':
        return str(0), 'N', 'do_augmentation'
    return None


def update_original_seed(item):
    try:
        classifier_model_dir = pathlib.Path(item['classifier']["S"])
        classifier_hparams = try_load_classifier_params(classifier_model_dir)
        seed = classifier_hparams['seed']
        return str(seed), "N", "original_training_seed"
    except Exception:
        return True, "NULL", "original_training_seed"


def update_fine_tuning_seed(item):
    return True, "NULL", "fine_tuning_seed"


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
    return classifier_model_dir_out.as_posix(), "S", "classifier"


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

    # update_classifier_db(client, table, get_classifier_source_env)
    # update_classifier_db(client, table, rename_classifier_model_dir)
    # update_classifier_db(client, table, update_original_seed)
    # update_classifier_db(client, table, update_do_augmentation)
    update_classifier_db(client, table, add_fine_tuning_take)
    # update_classifier_db(client, table, update_fine_tuning_seed)


if __name__ == '__main__':
    main()
