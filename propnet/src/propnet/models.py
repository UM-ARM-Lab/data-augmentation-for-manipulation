import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# noinspection PyPep8Naming
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from propnet.relations import construct_fully_connected_rel


class RelationEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RelationEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, n_relations, input_size]
        Returns:
            [batch_size, n_relations, output_size]
        """
        B, N, D = x.size()
        x = self.model(x.view(B * N, D))
        return x.view(B, N, self.output_size)


# noinspection PyPep8Naming
class ParticleEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticleEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, n_particles, input_size]
        Returns:
            [batch_size, n_particles, output_size]
        """
        B, N, D = x.size()
        x = self.model(x.view(B * N, D))
        return x.view(B, N, self.output_size)


# noinspection PyPep8Naming
class Propagator(nn.Module):
    def __init__(self, input_size, output_size, residual=False):
        super(Propagator, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.residual = residual

        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, res=None):
        """
        Args:
            x: [batch_size, n_relations/n_particles, input_size]
            res: residual
        Returns:
            [batch_size, n_relations/n_particles, output_size]
        """
        B, N, D = x.size()
        if self.residual:
            x = self.linear(x.view(B * N, D))
            x = self.relu(x + res.view(B * N, self.output_size))
        else:
            x = self.relu(self.linear(x.view(B * N, D)))

        return x.view(B, N, self.output_size)


# noinspection PyPep8Naming
class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticlePredictor, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: [batch_size, n_particles, input_size]
        Returns:
            [batch_size, n_particles, output_size]
        """
        B, N, D = x.size()
        x = x.view(B * N, D)
        x = self.linear_1(self.relu(self.linear_0(x)))
        return x.view(B, N, self.output_size)


# noinspection PyPep8Naming
class PropModule(pl.LightningModule):
    def __init__(self, params, input_dim, output_dim, batch=True, residual=False):

        super(PropModule, self).__init__()

        self.params = params
        self.batch = batch

        relation_dim = self.params['relation_dim']

        nf_particle = self.params['nf_particle']
        nf_relation = self.params['nf_relation']
        self.nf_effect = self.params['nf_effect']

        self.residual = residual

        # particle encoder
        self.particle_encoder = ParticleEncoder(input_dim, nf_particle, self.nf_effect)

        # relation encoder
        self.relation_encoder = RelationEncoder(2 * input_dim + relation_dim, nf_relation, nf_relation)

        # input: (1) particle encode (2) particle effect
        self.particle_propagator = Propagator(2 * self.nf_effect, self.nf_effect, self.residual)

        # input: (1) relation encode (2) sender effect (3) receiver effect
        self.relation_propagator = Propagator(nf_relation + 2 * self.nf_effect, self.nf_effect)

        # input: (1) particle effect
        self.particle_predictor = ParticlePredictor(self.nf_effect, self.nf_effect, output_dim)

    def forward(self, state, Rr, Rs, Ra, pstep, verbose: int = 0):

        if verbose:
            print('state size', state.size(), state.dtype)
            print('Rr size', Rr.size(), Rr.dtype)
            print('Rs size', Rs.size(), Rs.dtype)
            print('Ra size', Ra.size(), Ra.dtype)
            print('pstep', pstep)

        # calculate particle encoding
        particle_effect = Variable(torch.zeros((state.size(0), state.size(1), self.nf_effect))).to(self.device)

        # receiver_state, sender_state
        if self.batch:
            Rrp = torch.transpose(Rr, 1, 2)
            Rsp = torch.transpose(Rs, 1, 2)
            state_r = Rrp.bmm(state)
            state_s = Rsp.bmm(state)
        else:
            Rrp = Rr.t()
            Rsp = Rs.t()
            assert state.size(0) == 1
            state_r = Rrp.mm(state[0])[None, :, :]
            state_s = Rsp.mm(state[0])[None, :, :]

        if verbose:
            print('Rrp', Rrp.size())
            print('Rsp', Rsp.size())
            print('state_r', state_r.size())
            print('state_s', state_s.size())

        # particle encode
        particle_encode = self.particle_encoder(state)  # [b, n_objects, nf_particle]

        # calculate relation encoding
        relation_encode = self.relation_encoder(torch.cat([state_r, state_s, Ra], 2))  # [b, n_objects, nf_relation]

        if verbose:
            print("relation encode:", relation_encode.size())

        for i in range(pstep):
            if verbose:
                print("pstep", i)

            if self.batch:
                effect_r = Rrp.bmm(particle_effect)
                effect_s = Rsp.bmm(particle_effect)
            else:
                assert particle_effect.size(0) == 1
                effect_r = Rrp.mm(particle_effect[0])[None, :, :]
                effect_s = Rsp.mm(particle_effect[0])[None, :, :]

            # calculate relation effect
            relation_effect = self.relation_propagator(torch.cat([relation_encode, effect_r, effect_s], 2))

            if verbose:
                print("relation effect:", relation_effect.size())

            # calculate particle effect by aggregating relation effect
            if self.batch:
                effect_agg = Rr.bmm(relation_effect)
            else:
                assert relation_effect.size(0) == 1
                effect_agg = Rr.mm(relation_effect[0])[None, :, :]

            # calculate particle effect
            particle_effect = self.particle_propagator(
                torch.cat([particle_encode, effect_agg], 2),
                res=particle_effect)

            if verbose:
                print("particle effect:", particle_effect.size())

        pred = self.particle_predictor(particle_effect)

        if verbose:
            print("pred:", pred.size())

        return pred


# noinspection PyPep8Naming
class PropNet(pl.LightningModule):

    def __init__(self, params, scenario: ScenarioWithVisualization, residual=True):
        super().__init__()

        self.params = params
        self.scenario = scenario
        attr_dim = self.params['attr_dim']
        state_dim = self.params['state_dim']
        self.action_dim = self.params['action_dim']
        position_dim = self.params['position_dim']
        relation_dim = self.params['relation_dim']  # should match the last dim of Ra
        num_objects = self.params['num_objects']

        batch = True
        input_dim = attr_dim + state_dim + self.action_dim
        self.model = PropModule(params, input_dim, position_dim, batch, residual)

        Rr, Rs, Ra = construct_fully_connected_rel(num_objects, relation_dim, device=self.device)
        self.register_buffer("Rr", Rr)
        self.register_buffer("Rs", Rs)
        self.register_buffer("Ra", Ra)

    def forward(self, data, _, action=None):
        """
        Used only for fully observable case. Make only a 1-step prediction

        Args:
            data:
            _:
            action:

        Returns:

        """
        attr, state, Rr, Rs, Ra = data
        # Rr: [b, num_objects, num_relations], binary, 1 at [obj_i,rel_j] means object i is the receiver in relation j
        # Rs: [b, num_objects, num_relations], binary, 1 at [obj_i,rel_j] means object i is the sender in relation j
        # Ra: [b, num_objects^2, attr_dim] containing the relation attributes
        if action is not None:
            state = torch.cat([attr, state, action], 2)  # [b, num_objects, state+action]
        else:
            state = torch.cat([attr, state], 2)
        return self.model(state, Rr, Rs, Ra, self.params['pstep'], verbose=self.params['verbose'])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        batch_size = len(train_batch['filename'])
        Ra_batch, Rr_batch, Rs_batch = self.batch_relations(batch_size)

        loss = F.l1_loss(pred, label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        dt = torch.squeeze(val_batch['dt'], dim=-1)[:, 0]
        batch_size = len(val_batch['filename'])
        Ra_batch, Rr_batch, Rs_batch = self.batch_relations(batch_size)

        attr, states, actions = self.states_and_actions(val_batch, batch_size)
        gt_vel = states

        pred_state_t = states[:, 0]  # [b, n_objects, state_dim]
        for t in range(actions.shape[1]):
            action_t = actions[:, t]
            pred_vel_t = self.forward([attr, pred_state_t, Rr_batch, Rs_batch, Ra_batch], self.params['pstep'],
                                      action=action_t)
            # pred_vel_t: [b, n_objects, 3]
            pred_state_t = pred_state_t + dt * pred_vel_t

        loss = F.l1_loss(pred, gt_vel)
        self.log('val_loss', loss)

    def batch_relations(self, batch_size):
        Rr_batch = self.Rr[None, :, :].repeat(batch_size, 1, 1)
        Rs_batch = self.Rs[None, :, :].repeat(batch_size, 1, 1)
        Ra_batch = self.Ra[None, :, :].repeat(batch_size, 1, 1)
        return Ra_batch, Rr_batch, Rs_batch

    def states_and_actions(self, batch, batch_size):
        num_blocks = batch['num_blocks'][0, 0, 0]  # assumed fixed across batch/time
        time = batch['time_idx'].shape[1]
        attrs = []
        states = []
        actions = []

        # end-effector (more generally, the robot) is treated as an object in the system, so it has state.
        # It also has action and attribute=1
        # all of the object objects do not have action (zero value and attribute=0)
        robot_attr = torch.ones([batch_size, 1]).to(self.device)
        ee_pos = torch.squeeze(batch["jaco_arm/primitive_hand/tcp_pos"], 2)
        ee_quat = torch.squeeze(batch["jaco_arm/primitive_hand/orientation"], 2)
        ee_linear_vel = torch.squeeze(batch["jaco_arm/primitive_hand/linear_velocity"], 2)
        ee_angular_vel = torch.squeeze(batch["jaco_arm/primitive_hand/angular_velocity"], 2)
        ee_state = torch.cat([ee_pos, ee_quat, ee_linear_vel, ee_angular_vel], dim=-1)  # [b, T, 13]
        robot_action = batch['gripper_position']
        attrs.append(robot_attr)
        states.append(ee_state)
        actions.append(robot_action)

        # blocks
        for block_idx in range(num_blocks):
            block_attr = torch.zeros([batch_size, 1]).to(self.device)
            block_pos = torch.squeeze(batch[f"block{block_idx}/position"], 2)  # [b, T, 3]
            block_quat = torch.squeeze(batch[f"block{block_idx}/orientation"], 2)  # [b, T, 4]
            block_linear_vel = torch.squeeze(batch[f"block{block_idx}/linear_velocity"], 2)  # [b, T, 3]
            block_angular_vel = torch.squeeze(batch[f"block{block_idx}/angular_velocity"], 2)  # [b, T, 3]
            block_state = torch.cat([block_pos, block_quat, block_linear_vel, block_angular_vel], dim=-1)  # [b, T, 13]
            block_action = torch.zeros([batch_size, time - 1, self.action_dim]).to(self.device)
            attrs.append(block_attr)
            states.append(block_state)
            actions.append(block_action)
        attrs = torch.stack(attrs, dim=1)  # [b, n_objects, 1]
        states = torch.stack(states, dim=2)  # [b, n_objects, T, 13]
        actions = torch.stack(actions, dim=2)  # [b, n_objects, T-1, 13]
        return attrs, states, actions
