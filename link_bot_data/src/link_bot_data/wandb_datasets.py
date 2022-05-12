import pathlib
import shutil
from time import time, sleep

import wandb
from colorama import Fore
from wandb import CommError

from arc_utilities.filesystem_utils import rm_tree
from moonshine.filepath_tools import load_hjson


def wandb_save_dataset(dataset_dir: pathlib.Path, project: str, entity='armlab'):
    run_id = f'upload_data_{int(time())}'
    with wandb.init(project=project,
                    job_type="auto",
                    entity=entity,
                    id=run_id,
                    settings=wandb.Settings(silent='true')) as run:
        hparams = load_hjson(dataset_dir / 'hparams.hjson')
        print(Fore.GREEN + f"Saving dataset {dataset_dir.name}" + Fore.RESET)
        artifact = wandb.Artifact(dataset_dir.name, type="raw_data", metadata=hparams)
        artifact.add_dir(dataset_dir.as_posix())
        run.log_artifact(artifact)
    api = wandb.Api()
    while True:
        sleep(0.1)
        run = api.run(f"{entity}/{project}/{run_id}")
        if run.state == 'finished':
            run.delete()
            break


def get_dataset_with_version(dataset_dir: pathlib.Path, project, entity='armlab'):
    api = wandb.Api({'entity': entity})
    try:
        artifact = api.artifact(f"{project}/{dataset_dir.name}:latest")
        return f"{dataset_dir.name}-{artifact.version}"
    except CommError:
        return 'null'


def wandb_download_dataset(entity: str, project: str, dataset_name: str, version: str, outdir: pathlib.Path):
    api = wandb.Api()
    artifact = api.artifact(f'{entity}/{project}/{dataset_name}:{version}')
    artifact_dir = artifact.download()
    full_outdir = outdir / dataset_name
    shutil.copytree(artifact_dir, full_outdir)
    rm_tree(artifact_dir)
    return full_outdir
