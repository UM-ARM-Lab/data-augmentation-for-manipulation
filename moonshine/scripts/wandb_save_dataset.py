#!/usr/bin/env python
import argparse
import pathlib
from time import time, sleep

import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('project', type=str)
    parser.add_argument('--entity', type=str, default='armlab')

    args = parser.parse_args()

    run_id = f'upload_data_{int(time())}'
    with wandb.init(project=args.project,
                    job_type="auto",
                    entity=args.entity,
                    id=run_id,
                    settings=wandb.Settings(silent='true')) as run:
        artifact = wandb.Artifact(args.dataset_dir.name, type="raw_data")
        artifact.add_dir(args.dataset_dir)
        run.log_artifact(artifact)

    api = wandb.Api()
    while True:
        sleep(0.1)
        run = api.run(f"{args.entity}/{args.project}/{run_id}")
        if run.state == 'finished':
            run.delete()
            break


if __name__ == '__main__':
    main()
