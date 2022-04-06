import pathlib
from time import time, sleep

import wandb


def wandb_save_dataset(dataset_dir: pathlib.Path, project: str, entity='armlab'):
    run_id = f'upload_data_{int(time())}'
    with wandb.init(project=project,
                    job_type="auto",
                    entity=entity,
                    id=run_id,
                    settings=wandb.Settings(silent='true')) as run:
        artifact = wandb.Artifact(dataset_dir.name, type="raw_data")
        artifact.add_dir(dataset_dir.as_posix())
        run.log_artifact(artifact)
    api = wandb.Api()
    while True:
        sleep(0.1)
        run = api.run(f"{entity}/{project}/{run_id}")
        if run.state == 'finished':
            run.delete()
            break
