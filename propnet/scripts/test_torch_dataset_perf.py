#!/usr/bin/env python
import pathlib
from time import perf_counter

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from moonshine.torch_datasets_utils import repeat_dataset, my_collate
from propnet.train_test_propnet import get_num_workers
from state_space_dynamics.torch_dynamics_dataset import remove_keys, TorchMetaDynamicsDataset


def main():
    p = pathlib.Path("fwd_model_data/manual_val1_uncompressed2")
    batch_size = 512

    transform = transforms.Compose([
        # remove_keys('scene_msg'),
        remove_keys("scene_msg", "env", "sdf", "sdf_grad"),
    ])
    dataset = TorchMetaDynamicsDataset(p, transform=transform)
    dataset = repeat_dataset(dataset, 100)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        collate_fn=my_collate,
                        num_workers=get_num_workers(batch_size))

    t0 = perf_counter()
    for _ in tqdm(loader):
        pass
    print(perf_counter() - t0)

    # 12 workers, 159s
    # 13 minutes with 1 worker
    # 32 workers doesn't even work
    # 11 workers, 151s
    # after uncompressing all the data, with batch_size=16 we get, 163 <--- bad test, ignore this
    # uncompressed bs512, still 151s <--- bad test, ignore this
    # after removing the sdf, sdf_grad, env, and scene_msg from the data directly, 149s <--- bad test, ignore this
    # prefetch factor of 10, 152s
    # pin memory = True, still 148
    # with lru_cache, 75s
    # with un-compressed and lru_cache, 5s
    # with un-compressed no lru cache, 6s
    # uncompressed, full data dicts, no lru_cache, 62s
    # uncompressed, full data dicts, with lru_cache, 31s let's try that...

    # nothing helps !??!


if __name__ == '__main__':
    main()
