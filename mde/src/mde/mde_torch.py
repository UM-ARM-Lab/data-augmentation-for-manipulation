import pathlib
from typing import Dict, List

import gpytorch
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from botorch.models import HeteroskedasticSingleTaskGP
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import nn

import rospy
from autolab_core import YamlConfig
from link_bot_data.dataset_utils import add_predicted_hack, add_predicted
from link_bot_data.local_env_helper import LocalEnvHelper
from link_bot_data.visualization import DebuggingViz
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.grid_utils_np import environment_to_vg_msg
from link_bot_pycommon.load_wandb_model import load_model_artifact, load_gp_mde_from_cfg
from moonshine import get_local_environment_torch
from moonshine.make_voxelgrid_inputs_torch import VoxelgridInfo
from moonshine.robot_points_torch import RobotVoxelgridInfo
from moonshine.torch_and_tf_utils import remove_batch, add_batch
from moonshine.torch_utils import sequence_of_dicts_to_dict_of_tensors
from moonshine.torchify import torchify


def debug_vgs():
    return rospy.get_param("DEBUG_VG", False)


class MDE(pl.LightningModule):

    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        datset_params = self.hparams['dataset_hparams']
        data_collection_params = datset_params['data_collection_params']
        self.scenario = get_scenario(self.hparams.scenario, params=data_collection_params['scenario_params'])

        self.local_env_h_rows = self.hparams['local_env_h_rows']
        self.local_env_w_cols = self.hparams['local_env_w_cols']
        self.local_env_c_channels = self.hparams['local_env_c_channels']
        self.point_state_keys_pred = [add_predicted_hack(k) for k in self.hparams['point_state_keys']]

        conv_layers = []
        in_channels = 5
        for out_channels, kernel_size in self.hparams['conv_filters']:
            conv_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size))
            conv_layers.append(nn.LeakyReLU())
            conv_layers.append(nn.MaxPool3d(self.hparams['pooling']))
            in_channels = out_channels

        fc_layers = []
        state_desc = data_collection_params['state_description']
        action_desc = data_collection_params['action_description']
        state_size = sum([state_desc[k] for k in self.hparams.state_keys])
        action_size = sum([action_desc[k] for k in self.hparams.action_keys])

        conv_out_size = int(self.hparams['conv_filters'][-1][0] * np.prod(self.hparams['conv_filters'][-1][1]))
        prev_error_size = 1
        if self.hparams.get("use_prev_error", True):
            in_size = conv_out_size + 2 * state_size + action_size + prev_error_size
        else:
            in_size = conv_out_size + 2 * state_size + action_size
        use_drop_out = 'dropout_p' in self.hparams
        for hidden_size in self.hparams['fc_layer_sizes']:
            fc_layers.append(nn.Linear(in_size, hidden_size))
            fc_layers.append(nn.LeakyReLU())
            if use_drop_out:
                fc_layers.append(nn.Dropout(p=self.hparams.get('dropout_p', 0.0)))
            in_size = hidden_size

        final_hidden_dim = self.hparams['fc_layer_sizes'][-1]
        self.no_lstm = self.hparams.get('no_lstm', False)
        if not self.no_lstm:
            fc_layers.append(nn.LSTM(final_hidden_dim, self.hparams['rnn_size'], 1))

        self.conv_encoder = torch.nn.Sequential(*conv_layers)
        self.fc = torch.nn.Sequential(*fc_layers)

        if self.no_lstm:
            self.output_layer = nn.Linear(2 * final_hidden_dim, 1)
        else:
            self.output_layer = nn.Linear(final_hidden_dim, 1)

        self.debug = DebuggingViz(self.scenario, self.hparams.state_keys, self.hparams.action_keys)
        self.local_env_helper = LocalEnvHelper(h=self.local_env_h_rows, w=self.local_env_w_cols,
                                               c=self.local_env_c_channels,
                                               get_local_env_module=get_local_environment_torch)
        self.robot_info = RobotVoxelgridInfo(joint_positions_key=add_predicted_hack('joint_positions'))
        self.vg_info = VoxelgridInfo(h=self.local_env_h_rows,
                                     w=self.local_env_w_cols,
                                     c=self.local_env_c_channels,
                                     state_keys=self.point_state_keys_pred,
                                     jacobian_follower=self.scenario.robot.jacobian_follower,
                                     robot_info=self.robot_info,
                                     include_robot_geometry=True,
                                     scenario=self.scenario,
                                     )

        self.has_checked_training_mode = False

        self.val_accuracy = torchmetrics.Accuracy()

    def forward(self, inputs: Dict[str, torch.Tensor]):
        if not self.has_checked_training_mode:
            self.has_checked_training_mode = True
            print(f"Training Mode? {self.training}")

        if self.local_env_helper.device != self.device:
            self.local_env_helper.to(self.device)

        local_env, local_origin_point = self.get_local_env(inputs)

        batch_size, time = inputs['time_idx'].shape[0:2]
        voxel_grids = self.vg_info.make_voxelgrid_inputs(inputs, local_env, local_origin_point, batch_size, time,
                                                         viz=debug_vgs())

        if debug_vgs():
            b = 0
            for t in range(voxel_grids.shape[1]):
                self.debug.plot_pred_state_rviz(inputs, b, t, 'pred_inputs')
                for i in range(voxel_grids.shape[2]):
                    raster_dict = {
                        'env':          voxel_grids[b, t, i].cpu().numpy(),
                        'res':          inputs['res'][b].cpu().numpy(),
                        'origin_point': local_origin_point[b].cpu().numpy(),
                    }

                    self.scenario.send_occupancy_tf(raster_dict, parent_frame_id='robot_root',
                                                    child_frame_id='local_env_vg')
                    raster_msg = environment_to_vg_msg(raster_dict, frame='local_env_vg', stamp=rospy.Time.now())
                    self.debug.raster_debug_pubs[i].publish(raster_msg)

        states = {k: inputs[add_predicted_hack(k)] for k in self.hparams.state_keys}
        states_local_frame = self.scenario.put_state_local_frame_torch(states)
        states_local_frame_list = list(states_local_frame.values())
        actions = {k: inputs[k] for k in self.hparams.action_keys}
        all_but_last_states = {k: v[:, :-1] for k, v in states.items()}
        actions = self.scenario.put_action_local_frame(all_but_last_states, actions)
        padded_actions = [F.pad(v, [0, 0, 0, 1, 0, 0]) for v in actions.values()]

        states_robot_frame = self.scenario.put_state_robot_frame(states)
        states_robot_frame_list = list(states_robot_frame.values())

        flat_voxel_grids = voxel_grids.reshape(
            [-1, 5, self.local_env_h_rows, self.local_env_w_cols, self.local_env_c_channels])
        flat_conv_h = self.conv_encoder(flat_voxel_grids)
        conv_h = flat_conv_h.reshape(batch_size, time, -1)

        prev_pred_error = inputs['error'][:, 0].unsqueeze(-1).unsqueeze(-1)
        padded_prev_pred_error = F.pad(prev_pred_error, [0, 0, 0, 1, 0, 0])
        if self.hparams.get("use_prev_error", True):
            cat_args = [conv_h,
                        padded_prev_pred_error] + states_robot_frame_list + states_local_frame_list + padded_actions
        else:
            cat_args = [conv_h] + states_robot_frame_list + states_local_frame_list + padded_actions
        fc_in = torch.cat(cat_args, -1)

        if self.no_lstm:
            fc_out_h = self.fc(fc_in)
            out_h = fc_out_h.reshape(batch_size, -1)
            predicted_error = self.output_layer(out_h)
            predicted_error = predicted_error.squeeze(1)
        else:
            out_h, _ = self.fc(fc_in)
            # for every timestep's output, map down to a single scalar, the logit for accept probability
            predicted_errors = self.output_layer(out_h)
            predicted_error = predicted_errors[:, 1:].squeeze(1).squeeze(1)

        return predicted_error

    def get_local_env(self, inputs):
        batch_size = inputs['time_idx'].shape[0]
        state_0 = {k: inputs[add_predicted_hack(k)][:, 0] for k in self.hparams.point_state_keys}

        local_env_center = self.scenario.local_environment_center_differentiable_torch(state_0)
        local_env, local_origin_point = self.local_env_helper.get(local_env_center, inputs, batch_size)

        return local_env, local_origin_point

    def compute_loss(self, inputs: Dict[str, torch.Tensor], outputs):
        error_after = inputs['error'][:, 1]
        loss = F.mse_loss(outputs, error_after)
        return loss

    def training_step(self, train_batch: Dict[str, torch.Tensor], batch_idx):
        outputs = self.forward(train_batch)
        loss = self.compute_loss(train_batch, outputs)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch: Dict[str, torch.Tensor], batch_idx):
        pred_error = self.forward(val_batch)
        loss = self.compute_loss(val_batch, pred_error)
        true_error = val_batch['error'][:, 1]
        true_error_thresholded = true_error < self.hparams.error_threshold
        pred_error_thresholded = pred_error < self.hparams.error_threshold
        signed_loss = pred_error - true_error
        self.log('val_loss', loss)
        self.val_accuracy(pred_error_thresholded, true_error_thresholded)  # updates the metric
        self.log('pred_minus_true_error', signed_loss)
        return loss

    def validation_epoch_end(self, _):
        self.log('val_accuracy', self.val_accuracy.compute())  # logs the metric result/value
        # reset all metrics
        self.val_accuracy.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.hparams.learning_rate,
                                weight_decay=self.hparams.get('weight_decay', 0))


class GPRMDE():
    def __init__(self, **hparams):
        super().__init__()

    def load_model(self, model_fn, data_fn):
        self._model_heter = np.load(model_fn, allow_pickle=True)
        data = np.load(data_fn, allow_pickle=True)
        self.nonzero_std_dims = data["nonzero_std"]
        train_x = torch.from_numpy(data["datas_scaled"]).cuda()
        train_y = torch.from_numpy(data["labels_scaled"]).cuda()
        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
            ),
        )
        model = SingleTaskGP(train_X=train_x, train_Y=train_y, covar_module=covar_module).cuda()
        with torch.no_grad():
            observed_var = torch.pow(model.posterior(train_x).mean - train_y, 2)
        self._model_heter = HeteroskedasticSingleTaskGP(train_x, train_y, observed_var)
        state_dict = torch.load(model_fn)
        self._model_heter.load_state_dict(state_dict)
        self._likelihood = ExactMarginalLogLikelihood(self._model_heter.likelihood, self._model_heter)

    def eval(self):
        self._model_heter.eval()

    def predict(self, test_x):
        self._likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self._model_heter.posterior(test_x, observation_noise=True)
        return pred


class MDEConstraintChecker:

    def __init__(self, checkpoint):
        self.model: MDE = load_model_artifact(checkpoint, MDE, project='mde', version='best', user='armlab')
        self.model.eval()
        self.horizon = 2
        self.name = 'MDE'

    def check_constraint(self, environment: Dict, states_sequence: List[Dict], actions: List[Dict]):
        inputs = self.states_and_actions_to_torch_inputs(states_sequence, actions, environment)

        pred_error = remove_batch(self.model(add_batch(inputs)))
        return pred_error.detach().cpu().numpy()

    def states_and_actions_to_torch_inputs(self, states_sequence, actions, environment):
        states_dict = sequence_of_dicts_to_dict_of_tensors(states_sequence)
        actions_dict = sequence_of_dicts_to_dict_of_tensors(actions)
        inputs = {}
        environment = torchify(environment)
        inputs.update(environment)
        for action_key in self.model.hparams.action_keys:
            inputs[action_key] = actions_dict[action_key]
        for state_metadata_key in self.model.hparams.state_metadata_keys:
            inputs[state_metadata_key] = states_dict[state_metadata_key]
        for state_key in self.model.hparams.state_keys:
            planned_state_key = add_predicted(state_key)
            inputs[planned_state_key] = states_dict[state_key]
        if 'joint_names' in states_dict:
            inputs[add_predicted('joint_names')] = states_dict['joint_names']
        if 'joint_positions' in states_dict:
            inputs[add_predicted('joint_positions')] = states_dict['joint_positions']
        if 'error' in states_dict:
            inputs['error'] = states_dict['error'][:, 0]
        inputs['time_idx'] = torch.arange(2, dtype=torch.float32)
        return inputs


class GPMDEConstraintChecker:
    def __init__(self, config_path: pathlib.Path):
        self.cfg = YamlConfig(config_path)
        self.model, self.state_and_parameter_scaler, self.deviation_scaler = load_gp_mde_from_cfg(self.cfg, GPRMDE)
        self.model.eval()
        self.horizon = 2
        self.name = 'MDE'

    def states_and_actions_to_torch_inputs(self, states_sequence, action_sequence, _):
        state_keys = ['rope', 'right_gripper', 'left_gripper'] + ["joint_positions"]
        action_keys = ["left_gripper_position", "right_gripper_position"]
        states = []
        actions = []
        for state_data, action_data in zip(states_sequence, action_sequence):
            flattened_state = []
            for state_key in state_keys:
                data_pt = state_data[state_key]
                flattened_state.extend(data_pt.flatten())
            flattened_actions = []
            for action_key in action_keys:
                flattened_actions.extend(action_data[action_key].flatten())
            states.append(flattened_state)
            actions.append(flattened_actions)
        np_unscaled = np.hstack([np.vstack(states), np.vstack(actions)])[:, self.model.nonzero_std_dims]
        np_scaled = self.state_and_parameter_scaler.transform(np_unscaled).astype(np.float32)
        inputs = torch.from_numpy(np_scaled).cuda()
        return inputs

    def check_constraint(self, environment: Dict, states_sequence: List[Dict], actions: List[Dict]):
        inputs = self.states_and_actions_to_torch_inputs(states_sequence, actions, environment)
        pred = self.model.predict(inputs)

        pred_error_scaled, std_pred_scaled = pred.mean.detach().cpu().numpy(), np.sqrt(
            pred.variance.detach().cpu().numpy())
        pred_error_std_unscaled = ((
                                           std_pred_scaled ** 2) * self.deviation_scaler.var_) ** 0.5  # not used now but for reference
        d_hat = self.deviation_scaler.inverse_transform(pred_error_scaled)
        return d_hat[0]


if __name__ == '__main__':
    rospy.init_node("mde_torch_test")
    config_path = pathlib.Path("gp_mde_configs/gp_mde_test.yaml")
    c = GPMDEConstraintChecker(config_path)
    import pickle

    with open("mde_test_inputs.pkl", 'rb') as f:
        env, states, actions = pickle.load(f)
    outputs = c.check_constraint(env, states, actions)
    print(outputs)
