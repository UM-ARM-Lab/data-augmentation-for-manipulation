#!/usr/bin/env python
import argparse
import os
import pathlib

import wandb
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from dynamics_adaptation.dataset_statistics import metrics_funcs
from link_bot_data.wandb_datasets import get_dataset_with_version
from moonshine.numpify import numpify
from moonshine.torch_and_tf_utils import remove_batch
from moonshine.torch_datasets_utils import my_collate
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset, remove_keys
from state_space_dynamics.train_test_dynamics import load_udnn_model_wrapper


def main():
    os.environ["WANDB_SILENT"] = "true"

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('mode')

    args = parser.parse_args()

    transform = transforms.Compose([remove_keys("scene_msg")])
    dataset = TorchDynamicsDataset(args.dataset_dir, mode=args.mode, transform=transform)

    dataset_dir_versioned = get_dataset_with_version(args.dataset_dir, project='udnn', entity='armlab')

    run = wandb.init(project='datasets', entity='armlab', name=f'{dataset_dir_versioned}-{args.mode}')
    run.config['dataset_dir'] = args.dataset_dir
    run.config['dataset_dir_versioned'] = dataset_dir_versioned
    run.config['mode'] = args.mode

    model = load_udnn_model_wrapper('unadapted-b8s5s')
    model.eval()
    model.testing = True

    data = []
    columns = ['initial_model_error']
    for metric_func in metrics_funcs:
        columns.append(metric_func.__name__)

    loader = DataLoader(dataset, collate_fn=my_collate, num_workers=1)
    for example_batch in tqdm(loader):
        outputs_batch = model(example_batch)
        outputs = numpify(remove_batch(outputs_batch))
        example = remove_batch(example_batch)
        initial_model_error = model.scenario.classifier_distance(example, outputs)

        for t in range(10):
            example_t = dataset.index_time(example, t)
            row = [initial_model_error[t]]
            for metric_func in metrics_funcs:
                feature = metric_func(example_t)
                row.append(feature)
            data.append(row)

    table = wandb.Table(data=data, columns=columns)
    wandb.log({'dataset_statistics': table})


if __name__ == '__main__':
    main()
