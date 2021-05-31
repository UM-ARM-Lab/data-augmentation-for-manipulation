import time

from link_bot_pycommon.pycommon import paths_to_json


def setup_hparams(batch_size, dataset_dirs, seed, dataset_loader):
    return {
        'batch_size':           batch_size,
        'seed':                 seed,
        'datasets':             paths_to_json(dataset_dirs),
        'latest_training_time': int(time.time()),
        'dataset_hparams':      dataset_loader.hparams,
    }
