#!/usr/bin/env python
import pathlib
from time import perf_counter

from torch.utils.data import DataLoader
from tqdm import tqdm

from learn_invariance.new_dynamics_dataset import NewDynamicsDatasetLoader
from propnet.torch_dynamics_dataset import TorchDynamicsDataset
from propnet.train_test_propnet import get_num_workers


def main():
    p = pathlib.Path("fwd_model_data/blocks100-0_1635389142_a6cb0b3322_100")
    mode = 'all'
    batch_size = 32

    dataset = TorchDynamicsDataset(p, mode)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=get_num_workers(batch_size))

    t0 = perf_counter()
    for _ in tqdm(loader):
        pass
    print(perf_counter() - t0)

    # now my code...
    my_loader = NewDynamicsDatasetLoader([p])
    my_dataset = my_loader.batch(batch_size)

    t0 = perf_counter()
    for _ in tqdm(my_dataset):
        pass
    print(perf_counter() - t0)


if __name__ == '__main__':
    main()
