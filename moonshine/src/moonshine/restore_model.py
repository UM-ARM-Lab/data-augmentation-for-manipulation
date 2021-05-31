import tensorflow as tf
from colorama import Fore


def restore_model(model, path):
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=1)
    status = ckpt.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        print(Fore.CYAN + "Restored from {}".format(manager.latest_checkpoint) + Fore.RESET)
        if manager.latest_checkpoint:
            status.assert_nontrivial_match()
    else:
        raise RuntimeError(f"Failed to restore {manager.latest_checkpoint}!!!")