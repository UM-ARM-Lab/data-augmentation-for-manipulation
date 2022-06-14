import pathlib
from pathlib import Path

import numpy as np
import wandb


def load_model_artifact(checkpoint, model_class, project, version, user='armlab', **kwargs):
    local_ckpt_path = model_artifact_path(checkpoint, project, version, user)
    model = model_class.load_from_checkpoint(local_ckpt_path.as_posix(), from_checkpoint=checkpoint, **kwargs)
    return model


def load_gp_mde_from_cfg(cfg, model_class):
    run_path = cfg['run_path']
    cache_dir = cfg['cache_dir']
    root_path = Path(cache_dir) / run_path
    model_file_name = Path(root_path) / "validation_model.pkl"
    data_file_name = Path(root_path) / "other_data.pkl"
    deviation_scaler_fn = Path(root_path) / "deviation_scaler.pkl"
    state_and_parameter_scaler_fn = Path(root_path) / "state_and_parameter_scaler.pkl"
    files_to_restore = [model_file_name, deviation_scaler_fn, state_and_parameter_scaler_fn, data_file_name]
    for filepath in files_to_restore:
        wandb.restore(
            filepath.name,
            run_path=run_path,
            root=root_path
        )
    deviation_model = model_class()
    deviation_model.load_model(model_file_name, data_file_name)
    deviation_scaler = np.load(deviation_scaler_fn, allow_pickle=True)
    state_and_parameter_scaler = np.load(state_and_parameter_scaler_fn, allow_pickle=True)
    return deviation_model, state_and_parameter_scaler, deviation_scaler


def model_artifact_path(checkpoint, project, version, user='armlab'):
    artifact = get_model_artifact(checkpoint, project, user, version)

    # NOTE: this is much faster than letting .download() look up the manifest / cache etc...
    #  but may be incorrect if we modify the data without incrementing the version
    artifact_dir = pathlib.Path(artifact._default_root())
    if not artifact_dir.exists():
        artifact_dir = pathlib.Path(artifact.download())

    local_ckpt_path = artifact_dir / "model.ckpt"
    print(f"Found artifact {local_ckpt_path}")
    return local_ckpt_path


def resolve_latest_model_version(checkpoint, project, user):
    artifact = get_model_artifact(checkpoint, project, user, version='latest')
    return f'model-{checkpoint}:{artifact.version}'


def get_model_artifact(checkpoint, project, user, version):
    if ':' in checkpoint:
        checkpoint, version = checkpoint.split(':')
    if not checkpoint.startswith('model-'):
        checkpoint = 'model-' + checkpoint
    api = wandb.Api()
    artifact = api.artifact(f'{user}/{project}/{checkpoint}:{version}')
    return artifact
