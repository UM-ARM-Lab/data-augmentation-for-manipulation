import pathlib
from typing import List

import hjson
from tqdm import tqdm

from arc_utilities.filesystem_utils import rm_tree
from learn_invariance.new_dynamics_dataset import NewDynamicsDatasetLoader
from link_bot_data.dataset_utils import pkl_write_example
from link_bot_data.split_dataset import split_dataset
from moonshine.filepath_tools import load_params


def merge_dynamics_datasets_pkl(outdir: pathlib.Path, indirs: List[pathlib.Path]):
    if outdir.exists():
        q = input(f"Outdir {outdir} exists, do you want to overwrite it? [Y/n]")
        if q not in ['', 'y', 'Y']:
            print("aborting...")
            return
        rm_tree(outdir)

    outdir.mkdir()

    merged_dataset_loader = NewDynamicsDatasetLoader(indirs)
    merged_dataset = merged_dataset_loader.get_datasets(mode='all')

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

    for traj_idx, e in enumerate(tqdm(merged_dataset)):
        e['traj_idx'] = traj_idx
        if "full_filename" in e:
            e.pop("full_filename")
        if "filename" in e:
            e.pop("filename")
        pkl_write_example(outdir, e, traj_idx)

    split_dataset(outdir)
