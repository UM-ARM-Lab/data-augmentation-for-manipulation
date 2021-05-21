import time

from link_bot_pycommon.pycommon import paths_to_json


def setup_hparams(batch_size, dataset_dirs, seed, train_dataset, use_gt_rope):
    return {
        'batch_size':               batch_size,
        'seed':                     seed,
        'datasets':                 paths_to_json(dataset_dirs),
        'latest_training_time':     int(time.time()),
        'dynamics_dataset_hparams': train_dataset.hparams,
        'use_gt_rope':              use_gt_rope,
    }
