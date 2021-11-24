import argparse
import multiprocessing
import pathlib

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from moonshine.filepath_tools import load_hjson
from propnet.propnet_models import PropNet, get_batch_size
from propnet.torch_dynamics_dataset import TorchDynamicsDataset, remove_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)

    args = parser.parse_args()

    transform = transforms.Compose([
        remove_keys('filename', 'full_filename', 'joint_names', 'metadata'),
    ])

    train_dataset = TorchDynamicsDataset(args.dataset_dir, mode='train', transform=transform)

    train_loader = DataLoader(train_dataset, num_workers=multiprocessing.cpu_count())

    hparams = load_hjson(pathlib.Path("hparams/cylinders.hjson"))
    hparams['normalize_posvel'] = False  # turn off so we can gather the un-normalized data
    hparams['num_objects'] = train_dataset.params['data_collection_params']['num_objs'] + 1
    hparams['scenario'] = train_dataset.params['scenario']
    model = PropNet(hparams)

    posvels = []
    for batch in tqdm(train_loader):
        batch_size = get_batch_size(batch)

        attr, states = model.attr_and_states(batch, batch_size)

        for t in range(states.shape[1] - 1):
            state_t = states[:, t]
            pos_t = state_t[:model.hparams.position_dim]
            Rs, Rr, _ = model.scenario.propnet_rel(pos_t, model.hparams.num_objects, model.hparams.relation_dim,
                                                   is_close_threshold=model.hparams.is_close_threshold,
                                                   device=model.device)

            attr_state_t = torch.cat([attr, state_t], dim=-1)

            Rr_T = torch.transpose(Rr, 1, 2)
            Rs_T = torch.transpose(Rs, 1, 2)
            _, _, state_rel_posvel = model.model.relation_encoding(Rr_T, Rs_T, attr_state_t)

            state_rel_posvel = state_rel_posvel.squeeze()
            nonzero_state_rel_posvel_indices = torch.where(torch.any(state_rel_posvel > 0, dim=-1))
            nonzero_state_rel_posvel = state_rel_posvel[nonzero_state_rel_posvel_indices]

            posvels.append(nonzero_state_rel_posvel)

    posvels = torch.cat(posvels, dim=0)
    posvels = posvels.view([-1, posvels.shape[-1]])

    print(torch.std_mean(posvels, dim=0))


if __name__ == '__main__':
    main()
