import pathlib

import wandb


def load_model_artifact(checkpoint, model_class, project, version, user='armlab', **kwargs):
    local_ckpt_path = model_artifact_path(checkpoint, project, version, user)
    model = model_class.load_from_checkpoint(local_ckpt_path.as_posix(), **kwargs)
    return model


def model_artifact_path(checkpoint, project, version, user='armlab'):
    if ':' in checkpoint:
        checkpoint, version = checkpoint.split(':')

    if not checkpoint.startswith('model-'):
        checkpoint = 'model-' + checkpoint
    api = wandb.Api()
    artifact = api.artifact(f'{user}/{project}/{checkpoint}:{version}')
    artifact_dir = artifact.download()
    local_ckpt_path = pathlib.Path(artifact_dir) / "model.ckpt"
    print(f"Found artifact {local_ckpt_path}")
    return local_ckpt_path
