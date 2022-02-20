#!/usr/bin/env python
import pathlib
from time import perf_counter

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from learn_invariance.new_dynamics_dataset import NewDynamicsDatasetLoader
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset, remove_keys
from propnet.train_test_propnet import get_num_workers


def main():
    p = pathlib.Path("fwd_model_data/small_step3+vel")
    mode = 'all'
    batch_size = 32

    transform = transforms.Compose([
        remove_keys('filename', 'full_filename', 'joint_names', 'metadata'),
    ])
    dataset = TorchDynamicsDataset(p, mode, transform=transform)
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
