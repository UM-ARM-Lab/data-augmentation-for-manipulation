import pytorch_lightning as pl
import torch
from torch.autograd import Variable

from moonshine.torch_geometry import pairwise_squared_distances_self
from propnet.component_models import ParticlePredictor


class UDNN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)

        # input: (1) particle effect
        self.particle_predictor = ParticlePredictor(self.hparams.nf_effect,
                                                    self.hparams.nf_effect,
                                                    self.output_dim)

    def forward(self, state, Rr, Rs, Ra, pstep):
        # calculate particle encoding
        particle_effect = Variable(torch.zeros((state.size(0), state.size(1), self.hparams.nf_effect))).to(self.device)

        # receiver_state, sender_state
        Rr_T = torch.transpose(Rr, 1, 2)
        Rs_T = torch.transpose(Rs, 1, 2)

        # particle encode
        particle_encode = self.particle_encoder(state)  # [b, n_objects, nf_particle]

        # calculate relation encoding
        attr_r, attr_s, state_rel_posvel = self.relation_encoding(Rr_T, Rs_T, state)

        if self.hparams.get('normalize_posvel', False):
            state_rel_posvel = normalize(state_rel_posvel,
                                         mean=self.rel_posvel_mean,
                                         std=self.rel_posvel_std)

        relation_features = torch.cat([attr_r, attr_s, state_rel_posvel, Ra], dim=-1)
        relation_encode = self.relation_encoder(relation_features)  # [b, n_objects, nf_relation]

        for i in range(pstep):
            effect_r = Rr_T.bmm(particle_effect)
            effect_s = Rs_T.bmm(particle_effect)

            # calculate relation effect
            relation_effect = self.relation_propagator(torch.cat([relation_encode, effect_r, effect_s], dim=-1))

            # calculate particle effect by aggregating relation effect
            effect_agg = Rr.bmm(relation_effect)

            # calculate particle effect
            particle_effect = self.particle_propagator(torch.cat([particle_encode, effect_agg], 2), res=particle_effect)

        pred = self.particle_predictor(particle_effect)

        return pred

    def training_step(self, train_batch, batch_idx):
        gt_vel, gt_pos, pred_vel, pred_pos = self.forward(train_batch)
        loss = self.velocity_loss(gt_vel, pred_vel)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        gt_vel, gt_pos, pred_vel, pred_pos = self.forward(val_batch)
        loss = self.velocity_loss(gt_vel, pred_vel)
        self.log('val_loss', loss)
        max_error_vel = self.max_error_vel(gt_vel, pred_vel)
        self.log('max_error_vel', max_error_vel)
        max_error_pos = self.max_error_pos(gt_pos, pred_pos)
        self.log('max_error_pos', max_error_pos)
        mean_error_pos = self.mean_error_pos(gt_pos, pred_pos)
        self.log('mean_error_pos', mean_error_pos)

        radius = val_batch['radius'][0, 0, 0]
        dist_matrix = pairwise_squared_distances_self(pred_pos).sqrt()
        dist_to_nearest, _ = dist_matrix.min(dim=-1)

        self.log("penetration", self.penetration(radius, dist_to_nearest))
        self.log("spooky_action", self.spooky_action(radius, dist_to_nearest, pred_vel))
