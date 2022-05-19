import pathlib
from typing import List

import hjson
from tqdm import tqdm

from arc_utilities.filesystem_utils import rm_tree
from link_bot_data.split_dataset import write_mode
from link_bot_data.tf_dataset_utils import pkl_write_example
from moonshine.filepath_tools import load_params
from moonshine.my_torch_dataset import MyTorchDataset


def merge_dynamics_datasets_pkl(outdir: pathlib.Path, indirs: List[pathlib.Path]):
    if outdir.exists():
        q = input(f"Outdir {outdir} exists, do you want to overwrite it? [Y/n]")
        if q not in ['', 'y', 'Y']:
            print("aborting...")
            return
        rm_tree(outdir)

    outdir.mkdir()

    hparams_filename = 'hparams.hjson'
    path = indirs[0] / hparams_filename
    new_hparams_filename = outdir / hparams_filename
    # log this operation in the params!
    hparams = hjson.load(path.open('r'))
    hparams['created_by_merging'] = [str(indir) for indir in indirs]

    total_n_trajs = 0
    for indir in indirs:
        p = load_params(indir)
        total_n_trajs += p['n_trajs']
    hparams['n_trajs'] = total_n_trajs

    with new_hparams_filename.open('w') as new_hparams_file:
        hjson.dump(hparams, new_hparams_file, indent=2)
    print(path, '-->', new_hparams_filename)

    traj_idx = 0
    for mode in ['train', 'val', 'test']:
        files = []
        for dataset_dir in indirs:
            dataset = MyTorchDataset(dataset_dir, mode=mode, no_update_with_metadata=True)
            for e in tqdm(dataset):
                e['traj_idx'] = traj_idx
                _, full_metadata_filename = pkl_write_example(outdir, e, traj_idx)
                files.append(full_metadata_filename)
                traj_idx += 1
        write_mode(outdir, files, mode)
