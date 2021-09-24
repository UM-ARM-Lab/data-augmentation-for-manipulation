import pathlib

import numpy as np

from link_bot_data.new_base_dataset import NewBaseDatasetLoader
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)

if __name__ == '__main__':
    np.set_printoptions(linewidth=300)

    d = "classifier_data/untrained-aug-for-manual_0000-0003/"
    loader = NewBaseDatasetLoader([pathlib.Path(d)])
    dataset = loader.get_datasets(mode='all')
    dataset = dataset.take(32)
    dataset = dataset.shuffle()
    dataset = dataset.batch(8)
    dataset = dataset.shuffle()

    print("epoch 1")
    for e in dataset:
        print(e['traj_idx'])

    print("epoch 2")
    for e in dataset:
        print(e['traj_idx'])
