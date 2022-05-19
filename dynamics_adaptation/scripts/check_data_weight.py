#!/usr/bin/env python
import argparse

import torch

from link_bot_pycommon.load_wandb_model import load_model_artifact
from state_space_dynamics.mw_net import MWNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('example_indices', type=int, nargs='+')

    args = parser.parse_args()

    mwnet = load_model_artifact(args.checkpoint, MWNet, 'udnn', version='best', user='armlab', train_dataset=None)

    # print(f"{mwnet.hparams['max_example_idx']=}")
    # print(f"{mwnet.hparams['train_example_indices']=}")
    for example_idx in args.example_indices:
        u = mwnet.sample_weights[example_idx]
        w = torch.sigmoid(u)
        print(example_idx)
        print(f"\tUnnormalized weight: {u.detach().numpy():.3f}")
        print(f"\tNormalized weight:   {w.detach().numpy():.3f}")


if __name__ == '__main__':
    main()
