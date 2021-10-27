import argparse
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from progressbar import ProgressBar
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import PhysicsDataset, collate_fn
from models import PropNet
from utils import count_parameters, AverageMeter


@dataclass
class PropNetArgs:
    act_scale_min: Any
    action_dim: Any
    agg_method: Any
    attr_dim: Any
    batch_size: Any
    beta1: Any
    ckp_per_iter: Any
    dataf: Any
    dt: Any
    env: Any
    epoch: Any
    eval: Any
    eval_type: Any
    evalf: Any
    gen_data: Any
    gen_stat: Any
    history_window: Any
    len_seq: Any
    log_per_iter: Any
    lr: Any
    mpcf: Any
    n_epoch: Any
    n_particle: Any
    n_rollout: Any
    nf_effect: Any
    nf_particle: Any
    nf_relation: Any
    num_workers: Any
    outf: Any
    position_dim: Any
    pstep: Any
    pstep_decode: Any
    pstep_encode: Any
    relation_dim: Any
    resume_epoch: Any
    resume_iter: Any
    roll_step: Any
    scheduler_factor: Any
    scheduler_patience: Any
    src_dir: Any
    state_dim: Any
    time_step: Any
    train_valid_ratio: Any
    update_per_iter: Any
    verbose_data: Any
    verbose_model: Any


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pstep', type=int, default=3, help="propagation step")
    parser.add_argument('--pstep_encode', type=int, default=-1, help='propagation step for encoding, used for partial')
    parser.add_argument('--pstep_decode', type=int, default=-1, help='propagation step for decoding, used for partial')

    parser.add_argument('--n_rollout', type=int, default=0, help="number of rollout")
    parser.add_argument('--n_particle', type=int, default=0, help="number of particles")
    parser.add_argument('--time_step', type=int, default=0, help="time step per rollout")
    parser.add_argument('--dt', type=float, default=1. / 50., help="delta t between adjacent time step")

    parser.add_argument('--nf_relation', type=int, default=150, help="dim of hidden layer of relation encoder")
    parser.add_argument('--nf_particle', type=int, default=100, help="dim of hidden layer of object encoder")
    parser.add_argument('--nf_effect', type=int, default=100, help="dim of propagting effect")
    parser.add_argument('--agg_method', default='sum',
                        help='the method for aggregating the particle representations, sum|mean')

    parser.add_argument('--env', default='Rope', help="name of environment, Cradle|Rope|Box")
    parser.add_argument('--outf', default='files', help="name of log dir")
    parser.add_argument('--dataf', default='data', help="name of data dir")
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--gen_data', type=int, default=0, help="whether to generate new data")
    parser.add_argument('--gen_stat', type=int, default=0, help='whether to rengenerate statistics data')
    parser.add_argument('--train_valid_ratio', type=float, default=0.85, help="percentage of training data")

    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--update_per_iter', type=int, default=-1, help="update the network params every x iter")
    parser.add_argument('--log_per_iter', type=int, default=-1, help="print log every x iterations")
    parser.add_argument('--ckp_per_iter', type=int, default=-1, help="save checkpoint every x iterations")
    parser.add_argument('--eval', type=int, default=0, help="used for debugging")
    parser.add_argument('--n_epoch', type=int, default=1000)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--history_window', type=int, default=1, help='history window used for partial')
    parser.add_argument('--len_seq', type=int, default=2, help='train rollout length')
    parser.add_argument('--scheduler_factor', type=float, default=0.8)
    parser.add_argument('--scheduler_patience', type=float, default=0)

    parser.add_argument('--resume_epoch', type=int, default=-1)
    parser.add_argument('--resume_iter', type=int, default=-1)

    parser.add_argument('--verbose_data', type=int, default=0, help="print debug information during data loading")
    parser.add_argument('--verbose_model', type=int, default=0, help="print debug information during model forwarding")
    parser.add_argument('--attr_dim', type=int, default=0)
    parser.add_argument('--state_dim', type=int, default=0)
    parser.add_argument('--action_dim', type=int, default=0)
    parser.add_argument('--relation_dim', type=int, default=0)

    args = parser.parse_args()

    # Rope
    args.pn_mode = 'full'
    args.n_rollout = 5000
    args.time_step = 100
    args.n_particle = 15
    args.dt = 1. / 50.

    # attr [moving, fixed, radius]
    args.attr_dim = 3

    # state [x, y, xdot, ydot]
    args.state_dim = 4
    args.position_dim = 2

    # action [act_x, act_y]
    args.action_dim = 2

    # relation [collision, onehop, bihop]
    args.relation_dim = 3

    args.batch_size = 32
    args.log_per_iter = 2000
    args.ckp_per_iter = 10000
    args.update_per_iter = 1
    args.len_seq = 2
    args.scheduler_patience = 2
    args.outf = 'dump_Rope/' + args.outf

    # make names for log dir and data dir
    args.outf = args.outf + '_' + args.env
    if args.env == 'Box':
        args.outf += '_pstep_' + str(args.pstep_encode) + '_' + str(args.pstep_decode)
        args.outf += '_hisWindow_' + str(args.history_window)
        args.outf += '_lenSeq_' + str(args.len_seq)
    else:
        args.outf += '_pstep_' + str(args.pstep)

    args.dataf = 'data/' + args.dataf + '_' + args.env
    os.system('mkdir -p ' + args.outf)
    os.system('mkdir -p ' + args.dataf)


def generate_data(args: PropNetArgs):
    # generate data
    datasets = {phase: PhysicsDataset(args, phase) for phase in ['train', 'valid']}
    for phase in ['train', 'valid']:
        if args.gen_data:
            datasets[phase].gen_data()
        else:
            datasets[phase].load_data()

    use_gpu = torch.cuda.is_available()

    if args.pn_mode == 'full':
        dataloaders = {x: torch.utils.data.DataLoader(
            datasets[x], batch_size=args.batch_size,
            shuffle=True if x == 'train' else False,
            num_workers=args.num_workers)
            for x in ['train', 'valid']}
    elif args.pn_mode == 'partial':
        dataloaders = {x: torch.utils.data.DataLoader(
            datasets[x], batch_size=args.batch_size,
            shuffle=True if x == 'train' else False,
            num_workers=args.num_workers,
            collate_fn=collate_fn)
            for x in ['train', 'valid']}

    # define model network
    model = PropNet(args, residual=True, use_gpu=use_gpu)

    # print model #params
    print("model #params: %d" % count_parameters(model))

    # if resume from a pretrained checkpoint
    if args.resume_epoch >= 0:
        model_path = os.path.join(args.outf, 'net_epoch_%d_iter_%d.pth' % (args.resume_epoch, args.resume_iter))
        print("Loading saved ckp from %s" % model_path)
        model.load_state_dict(torch.load(model_path))


def construct_relations(n_particle, relation_dim):
    pass


def train(model, args: PropNetArgs, dataloaders):
    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, 'min',
                                  factor=args.scheduler_factor,
                                  patience=args.scheduler_patience,
                                  verbose=True)

    model = model.cuda()

    st_epoch = args.resume_epoch if args.resume_epoch > 0 else 0
    best_valid_loss = np.inf

    Rr, Rs, Ra = construct_relations(args.n_particle, args.relation_dim)

    for epoch in range(st_epoch, args.n_epoch):

        phases = ['train', 'valid'] if args.eval == 0 else ['valid']

        for phase in phases:

            model.train(phase == 'train')

            meter_loss = AverageMeter()

            bar = ProgressBar(max_value=len(dataloaders[phase]))

            loader = dataloaders[phase]

            for i, data in bar(enumerate(loader)):

                with torch.set_grad_enabled(phase == 'train'):
                    assert len(data) == 4
                    attr, state, action, label = [x.cuda() for x in data]

                    bs = attr.size(0)
                    Rr_batch = Rr[None, :, :].repeat(bs, 1, 1)
                    Rs_batch = Rs[None, :, :].repeat(bs, 1, 1)
                    Ra_batch = Ra[None, :, :].repeat(bs, 1, 1)

                    pred = model([attr, state, Rr_batch, Rs_batch, Ra_batch], args.pstep, action=action)

                    loss = F.l1_loss(pred, label)

                # prediction loss
                meter_loss.update(loss.item(), n=bs)

                if phase == 'train':
                    if i % args.update_per_iter == 0:
                        # update parameters every args.update_per_iter
                        if i != 0:
                            loss_acc /= args.update_per_iter
                            optimizer.zero_grad()
                            loss_acc.backward()
                            optimizer.step()
                        loss_acc = loss
                    else:
                        loss_acc += loss

                if i % args.log_per_iter == 0:
                    log = '%s [%d/%d][%d/%d] Loss: %.6f (%.6f)' % (
                        phase, epoch, args.n_epoch, i, len(loader), loss.item(), meter_loss.avg)

                    print()
                    print(log)

                if phase == 'train' and i % args.ckp_per_iter == 0:
                    torch.save(model.state_dict(), '%s/net_epoch_%d_iter_%d.pth' % (args.outf, epoch, i))

            log = '%s [%d/%d] Loss: %.4f, Best valid: %.4f' % (
                phase, epoch, args.n_epoch, meter_loss.avg, best_valid_loss)
            print(log)

            if phase == 'valid' and not args.eval:
                scheduler.step(meter_loss.avg)
                if meter_loss.avg < best_valid_loss:
                    best_valid_loss = meter_loss.avg
                    torch.save(model.state_dict(), '%s/net_best.pth' % (args.outf))


if __name__ == '__main__':
    main()
