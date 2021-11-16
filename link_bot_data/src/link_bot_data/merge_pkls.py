import pathlib
import pickle
import shutil
from typing import List

import hjson

from link_bot_data.dataset_utils import index_to_filename
from link_bot_data.split_dataset import split_dataset
from moonshine.filepath_tools import load_params


def merge_dynamics_datasets_pkl(outdir: pathlib.Path, indirs: List[pathlib.Path], dry_run: bool = False, quiet=False):
    outdir.mkdir(exist_ok=True)

    if not dry_run:
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
        if not quiet:
            print(path, '-->', new_hparams_filename)

    pkl_files = []
    pkl_gz_files = []
    for in_dir in indirs:
        pkl_files.extend(in_dir.glob("*.pkl"))
        pkl_gz_files.extend(in_dir.glob("*.pkl.gz"))

    traj_idx = 0
    for i, file in enumerate(sorted(pkl_files)):
        path = pathlib.Path(file)
        new_pkl_filename = index_to_filename('.pkl', traj_idx)
        new_pkl_gz_filename = index_to_filename('.pkl.gz', traj_idx)
        new_pkl_path = pathlib.Path(outdir) / new_pkl_filename
        traj_idx += 1
        if not quiet:
            print(path, '-->', new_pkl_path)
        if not dry_run:
            with path.open("rb") as f:
                pkl = pickle.load(f)
            new_pkl = pkl
            new_pkl['data'] = new_pkl_gz_filename
            with new_pkl_path.open("wb") as f:
                pickle.dump(new_pkl, f)

    traj_idx = 0
    for i, file in enumerate(sorted(pkl_gz_files)):
        path = pathlib.Path(file)
        new_pkl_gz_filename = index_to_filename('.pkl.gz', traj_idx)
        new_pkl_gz_path = pathlib.Path(outdir) / new_pkl_gz_filename
        traj_idx += 1
        if not quiet:
            print(path, '-->', new_pkl_gz_path)
        if not dry_run:
            shutil.copyfile(path, new_pkl_gz_path)

    if not dry_run:
        split_dataset(outdir)
