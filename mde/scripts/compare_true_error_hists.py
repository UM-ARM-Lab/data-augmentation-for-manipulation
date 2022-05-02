import argparse
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

from arc_utilities import ros_init
from mde.torch_mde_dataset import TorchMDEDataset
from moonshine.torch_datasets_utils import my_collate


@ros_init.with_ros("compare_true_error_hists")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('--mode', type=str, default='all')

    args = parser.parse_args()

    plt.style.use('slides')

    modes = ['train', 'test']

    for mode in modes:
        datasets = [TorchMDEDataset(dataset_dir, mode=mode, only_metadata=True) for dataset_dir in args.dataset_dirs]

        plt.figure(figsize=(10, 8))
        ax = plt.gca()

        all_errors = []
        for dataset_i in datasets:
            true_errors_i = []
            loader = DataLoader(dataset_i, collate_fn=my_collate, batch_size=16, shuffle=False)
            for batch in tqdm(loader):
                true_error = batch['error'][:, 1]
                true_errors_i.extend(true_error.detach().cpu().numpy().tolist())
            all_errors.append(true_errors_i)

        for true_errors_i, dataset_dir in zip(all_errors, args.dataset_dirs):
            sns.kdeplot(ax=ax, x=true_errors_i, label=dataset_dir.name, alpha=0.5)

        ax.set_xlabel("true error")
        ax.set_ylabel("count")
        ax.set_xlim([0, 1])
        ax.set_title(f"Error distribution ({mode})")
        plt.legend()
        plt.savefig(f"results/compare_mde_true_error_hists_mode={mode}.png")

    plt.show()


if __name__ == '__main__':
    main()
