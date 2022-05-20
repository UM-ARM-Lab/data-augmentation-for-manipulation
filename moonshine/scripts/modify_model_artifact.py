import pathlib

from pytorch_lightning import Trainer

import wandb
from arc_utilities.algorithms import nested_dict_update
from link_bot_pycommon.load_wandb_model import get_model_artifact
from state_space_dynamics.mw_net import MWNet


def main():
    checkpoint = "car4alt_train2_meta-2l3im"
    old_best_artifact = get_model_artifact(checkpoint, 'udnn', 'armlab', 'best')
    old_latest_artifact = get_model_artifact(checkpoint, 'udnn', 'armlab', 'latest')
    artifact_dir = old_best_artifact.download()
    local_ckpt_path = pathlib.Path(artifact_dir) / "model.ckpt"
    print(f"Found {local_ckpt_path.as_posix()}")

    model = MWNet.load_from_checkpoint(local_ckpt_path.as_posix(), train_dataset=None)
    hparams_update = {
        'dataset_hparams': {
            'data_collection_params': {
                'max_distance_gripper_can_move': 0.1,
                'res':                           0.02,
            }
        }
    }
    nested_dict_update(model.hparams, hparams_update)

    trainer = Trainer()
    trainer.model = model
    trainer.save_checkpoint(local_ckpt_path)  # overwrite the old local ckpt file

    new_artifact = wandb.Artifact(name=f'model-{checkpoint}', type='model')
    new_artifact.metadata = old_best_artifact.metadata
    new_artifact.add_file(local_ckpt_path.as_posix())
    new_artifact.save('udnn', settings={'entity': 'armlab'})
    new_artifact.wait()
    new_artifact.aliases.append('best')
    print(f"Version: {new_artifact.version}")

    # The previous save op will automatically make it the latest version, but we don't want that. So now we restore
    # the 'latest' alias
    old_latest_artifact.aliases.append("latest")
    old_latest_artifact.save()
    old_latest_artifact.wait()

    # delete it after saving
    local_ckpt_path.unlink()


if __name__ == '__main__':
    main()
