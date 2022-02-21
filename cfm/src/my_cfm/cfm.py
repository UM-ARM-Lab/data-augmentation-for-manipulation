from copy import deepcopy
from typing import Dict

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.models import Sequential

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.constants import KINECT_MAX_DEPTH
from moonshine.image_augmentation import augment, resize_image_sequence, flatten_batch_and_sequence, unflatten_batch_and_sequence
from moonshine.loss_utils_tf import loss_on_dicts
from moonshine.tensorflow_utils import vector_to_dict
from moonshine.my_keras_model import MyKerasModel
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction
from state_space_dynamics.base_filter_function import BaseFilterFunction


class CFM(MyKerasModel):

    def __init__(self, hparams: Dict, batch_size: int, scenario):
        super().__init__(hparams=hparams, batch_size=batch_size)
        self.scenario = scenario

        self.obs_keys = self.hparams['obs_keys']
        self.state_keys = self.hparams['state_keys']
        self.action_keys = self.hparams['action_keys']
        self.observation_feature_keys = self.hparams['observation_feature_keys']
        self.use_observation_feature_loss = self.hparams['use_observation_feature_loss']
        self.use_cfm_loss = self.hparams['use_cfm_loss']

        self.encoder = Encoder(hparams, batch_size, scenario)
        self.dynamics = LocallyLinearPredictor(hparams, batch_size, scenario)
        self.observer = Observer(hparams, batch_size, scenario)

        self.encoder.trainable = hparams['encoder_trainable']

        self.tf_rng = tf.random.Generator.from_seed(1)

        # check for NaNs, because the Kinect uses NaN to indicate "missing" data (usually out of range)
        # tf.debugging.enable_check_numerics()

    def apply_gradients(self, tape, train_element, train_outputs, losses):
        train_batch_loss = losses['loss']
        variables = self.trainable_variables
        gradients = tape.gradient(train_batch_loss, variables)
        # g can be None if there are parts of the network not being trained, i.e. the observer with there are no obs. feats.
        valid_grads_and_vars = [(g, v) for (g, v) in zip(gradients, variables) if g is not None]
        # clip for training stability
        valid_grads_and_vars = [(tf.clip_by_norm(g, 1), v) for (g, v) in valid_grads_and_vars]
        self.optimizer.apply_gradients(valid_grads_and_vars)
        return {

        }

    def get_positive_pairs(self, example):
        obs = {k: example[k][:, :-1] for k in self.obs_keys}
        obs_pos = {k: example[k][:, 1:] for k in self.obs_keys}
        return obs, obs_pos

    def preprocess_no_gradient(self, example, training: bool):
        example = self.encoder.preprocess_no_gradient(example, training)
        example = self.dynamics.preprocess_no_gradient(example, training)
        return example

    # @tf.function
    def call(self, example, training=False, **kwargs):
        # NOTE: we are right now only doing 1-step prediction, and the perception predictions are 0-step
        observation, observation_pos = self.get_positive_pairs(example)

        # forward pass
        z, z_pos = self.encoder(observation), self.encoder(observation_pos)  # z is b x z_dim
        pred_inputs = {
            'z': z['z'],
            'z_pos': z_pos['z'],
        }
        for k in self.action_keys:
            pred_inputs[k] = example[k]
        z_seq = self.dynamics(pred_inputs)
        y_seq = self.observer(pred_inputs)

        output = {
            'z_pos': z_pos['z'],
        }
        output.update(z_seq)
        output.update(y_seq)
        return output

    def compute_loss(self, example, outputs):
        loss = 0
        if self.use_cfm_loss:
            cfm_loss = self.cfm_loss(outputs)
            loss += cfm_loss
        if self.use_observation_feature_loss:
            observation_feature_loss = self.observer_loss(example, outputs)
            loss += observation_feature_loss
        if self.hparams['use_stability_loss']:
            stability_loss = self.dynamics.stability_loss(example, outputs)
            loss += stability_loss

        return {
            'loss': loss
        }

    def compute_metrics(self, dataset_element, outputs):
        neg_dists, pos_dists = self.contrastive_distances(outputs)
        dists_matrix = tf.concat((neg_dists, pos_dists), axis=1)  # b x b+1
        softmax_dists_matrix = tf.nn.softmax(dists_matrix)
        softmax_dists_matrix = softmax_dists_matrix / tf.reduce_max(softmax_dists_matrix)
        dists_image = softmax_dists_matrix[tf.newaxis, :, :, tf.newaxis]
        metrics = {
            'min_neg_d': tf.reduce_min(neg_dists),
            'mean_neg_d': tf.reduce_mean(neg_dists),
            'max_neg_d': tf.reduce_max(neg_dists),
            'min_pos_d': tf.reduce_min(pos_dists),
            'mean_pos_d': tf.reduce_mean(pos_dists),
            'max_pos_d': tf.reduce_max(pos_dists),
            'dists_matrix': dists_image,
        }
        metrics.update(self.dynamics.compute_metrics(dataset_element, outputs))
        metrics.update(self.encoder.compute_metrics(dataset_element, outputs))
        metrics.update(self.observer.compute_metrics(dataset_element, outputs))
        return metrics

    def observer_loss(self, example, outputs):
        observation_feature_true = {k: example[k][:, :-1] for k in self.observation_feature_keys}
        observation_feature_predictions = {k: outputs[k] for k in self.observation_feature_keys}
        observation_feature_loss = loss_on_dicts(tf.keras.losses.mse,
                                                 dict_true=observation_feature_true,
                                                 dict_pred=observation_feature_predictions)
        return observation_feature_loss

    def cfm_loss(self, outputs):
        neg_dists, pos_dists = self.contrastive_distances(outputs)
        batch_size = outputs['z'].shape[0]
        mask = tf.one_hot(tf.range(batch_size), batch_size, on_value=1e12)
        neg_dists_masked = neg_dists - mask  # b x b+1
        dists = tf.concat((neg_dists_masked, pos_dists), axis=1)  # b x b+1
        log_probabilities = tf.nn.log_softmax(logits=dists, axis=-1)  # b x b+1

        loss = -tf.reduce_mean(log_probabilities[:, -1])  # Get last column which is the true pos sample
        return loss

    def contrastive_distances(self, outputs):
        z = outputs['z'][:, 0]
        z_pos = outputs['z_pos'][:, 0]
        z_next = outputs['z'][:, 1]  # this assumes single transitions
        batch_size = z_pos.shape[0]
        tiled_z = tf.tile(z[tf.newaxis, :], [batch_size, 1, 1])
        tiled_z_next = tf.tile(z_next[:, tf.newaxis], [1, batch_size, 1])
        tiled_z_next2 = tf.transpose(tiled_z_next, [1, 0, 2])

        # NOTE: z could be z_next here? probably doesn't matter, but using z_next makes the distances matrix symmetric
        if self.hparams['use_z_next_as_neg']:
            neg_dists = tf.squeeze(self.scenario.cfm_distance(tiled_z_next2, tiled_z_next), axis=2)
        else:
            neg_dists = tf.squeeze(self.scenario.cfm_distance(tiled_z, tiled_z_next), axis=2)

        # Subtracting a large positive values should make the loss for diagonal elements be 0
        # which means we don't encourage separating the representation of z from z_next
        pos_dists = self.scenario.cfm_distance(z_pos, z_next)
        return neg_dists, pos_dists


class Encoder(MyKerasModel):

    def __init__(self, hparams: Dict, batch_size: int, scenario: ExperimentScenario):
        super().__init__(hparams=hparams, batch_size=batch_size)
        self.scenario = scenario
        self.obs_keys = self.hparams['obs_keys']
        self.state_keys = self.hparams['state_keys']
        self.action_keys = self.hparams['action_keys']
        self.image_key = 'rgbd'

        self.z_dim = self.hparams['z_dim']
        self.model = Sequential([
            layers.Conv2D(filters=64, kernel_size=3),
            layers.LeakyReLU(0.2),
            layers.Conv2D(filters=64, kernel_size=4, strides=2),
            layers.LeakyReLU(0.2),
            layers.Conv2D(filters=64, kernel_size=3, strides=1),
            layers.LeakyReLU(0.2),
            layers.Conv2D(filters=128, kernel_size=4, strides=2),
            layers.LeakyReLU(0.2),
            layers.Conv2D(filters=256, kernel_size=4, strides=2),
            layers.LeakyReLU(0.2),
            layers.Conv2D(filters=256, kernel_size=4, strides=2),
            layers.LeakyReLU(0.2),
        ])

        # self.model = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=(90, 160, 4))
        # self.model.trainable = False

        self.out = layers.Dense(self.z_dim)

        self.tf_rng = tf.random.Generator.from_seed(1)

    def normalize(self, example: Dict):
        # this copy is required otherwise calling this inside @tf.function complains about modifying python arguments
        new_example = deepcopy(example)

        # normalize to 0-1
        min_value = tf.constant([[[0, 0, 0, 0]]], dtype=tf.float32)
        max_value = tf.constant([[[255, 255, 255, KINECT_MAX_DEPTH]]], dtype=tf.float32)
        normalized_observation = (example[self.image_key] - min_value) / (max_value - min_value)
        new_example[self.image_key] = normalized_observation

        return new_example

    def preprocess_no_gradient(self, example, training: bool):
        example = self.normalize(example)
        image_h = self.scenario.IMAGE_H
        image_w = self.scenario.IMAGE_W

        # always resize
        example[self.image_key] = resize_image_sequence(example[self.image_key], image_h, image_w)

        ## BEGIN DEBUG
        # show_image_grid(example[self.image_key])
        ## END DEBUG

        if self.hparams['image_augmentation'] and training:
            augmented = augment(example[self.image_key], image_h, image_w, generator=self.tf_rng)
            example[self.image_key] = augmented
        return example

    # @tf.function
    def call(self, observation: Dict, **kwargs):
        # NOTE: currently this encoder only handles one image as the observation input
        o = observation[self.image_key]

        o_one_batch, original_batch_dims = flatten_batch_and_sequence(o)
        h_one_batch = self.model(o_one_batch)
        h = unflatten_batch_and_sequence(h_one_batch, original_batch_dims)

        # NOTE: [:-3] gets all but the last 3 dimensions, which are the H, W, and C of the tensor
        # doing this specifically allows x to have multiple "batch" dimensions,
        # which is useful to treating [batch, time, ...] as all just batch dimensions
        h = tf.reshape(h, h.shape.as_list()[:-3] + [-1])
        z = self.out(h)
        outputs = {
            'z': z
        }
        outputs.update({k: observation[k] for k in self.obs_keys})
        return outputs


class LocallyLinearPredictor(MyKerasModel):

    def __init__(self, hparams: Dict, batch_size: int, scenario: ExperimentScenario):
        super().__init__(hparams=hparams, batch_size=batch_size)
        self.state_keys = self.hparams['state_keys']
        self.action_keys = self.hparams['action_keys']
        self.obs_keys = self.hparams['obs_keys']
        self.dynamics_type = self.hparams['dynamics_type']
        self.scenario = scenario

        self.z_dim = self.hparams['z_dim']
        # sum([self.hparams['dynamics_dataset_hparams']['action_description'][k] for k in self.action_keys])
        self.u_dim = 6

        my_layers = []
        for h in self.hparams['dynamics_fc_layer_sizes']:
            my_layers.append(layers.Dense(h))
            my_layers.append(layers.LeakyReLU(0.2))

        if self.dynamics_type == 'locally-linear':
            self.a_out = layers.Dense(self.z_dim * self.z_dim, activation='tanh')
        elif self.dynamics_type == 'quasi-static':
            self.a_out = layers.Dense(self.z_dim * self.z_dim, activation='tanh')
            self.b_out = layers.Dense(self.z_dim * self.u_dim, activation='tanh')
        elif self.dynamics_type == 'mlp':
            my_layers.append(layers.Dense(self.z_dim, activation=None))

        self.model = Sequential(my_layers)

    def preprocess_no_gradient(self, example, training: bool):
        # this copy is required otherwise calling this inside @tf.function complains about modifying python arguments
        new_example = deepcopy(example)  # https://stackoverflow.com/questions/3975376
        local_action = self.scenario.put_action_local_frame(example, example)
        new_example.update(local_action)

        return new_example

    def stability_loss(self, example, outputs):
        stability_loss = tf.reduce_mean(tf.nn.relu(outputs['max_eig'] - 1))
        return stability_loss

    # @tf.function
    def call(self, inputs, **kwargs):
        a = tf.concat([inputs[k] for k in self.action_keys], axis=-1)

        z = inputs['z']
        x = tf.concat((z, a), axis=-1)
        out = self.model(x)

        if self.dynamics_type == 'locally-linear':
            a_params = self.a_out(out)
            a_matrix = tf.reshape(a_params, x.shape.as_list()[:-1] + [self.z_dim, self.z_dim])
            z_col = tf.expand_dims(z, axis=-1)
            z_next = tf.linalg.matmul(a_matrix, z_col)
            z_next = tf.squeeze(z_next, axis=-1)
            eigs = tf.squeeze(tf.linalg.eigvals(a_matrix), axis=1)
            max_eig = tf.math.reduce_max(tf.math.abs(eigs), axis=1)
            z_seq = tf.concat([z, z_next], axis=1)
            return {
                'z': z_seq,
                'max_eig': max_eig,
            }
        elif self.dynamics_type == 'quasi-static':
            a_params = self.a_out(out)
            b_params = self.b_out(out)
            identity = tf.eye(self.z_dim)[tf.newaxis, tf.newaxis]
            a_matrix = tf.reshape(a_params, x.shape.as_list()[:-1] + [self.z_dim, self.z_dim]) + identity
            b_matrix = tf.reshape(b_params, x.shape.as_list()[:-1] + [self.z_dim, self.u_dim]) * 100.0
            z_col = tf.expand_dims(z, axis=-1)
            a_col = tf.expand_dims(a, axis=-1)
            # z_next = tf.linalg.matmul(a_matrix, z_col) + tf.linalg.matmul(b_matrix, a_col)
            z_next = z_col + tf.linalg.matmul(b_matrix, a_col)
            z_next = tf.squeeze(z_next, axis=-1)
            eigs = tf.squeeze(tf.linalg.eigvals(a_matrix), axis=1)
            max_eig = tf.math.reduce_max(tf.math.abs(eigs), axis=1)
            z_seq = tf.concat([z, z_next], axis=1)
            return {
                'z': z_seq,
                'max_eig': max_eig,
            }
        elif self.dynamics_type == 'mlp':
            z_next = self.model(x)
            z_seq = tf.concat([z, z_next], axis=1)
            return {
                'z': z_seq,
            }
        else:
            raise NotImplementedError(f"unimplemented dynamics type {self.dynamics_type}")

    def compute_loss(self, dataset_element, outputs):
        raise NotImplementedError()

    def compute_metrics(self, dataset_element, outputs):
        return {
        }


class Observer(MyKerasModel):

    def __init__(self, hparams: Dict, batch_size: int, scenario: ExperimentScenario):
        super().__init__(hparams=hparams, batch_size=batch_size)
        self.state_keys = self.hparams['state_keys']
        self.observation_features_keys = self.hparams['observation_feature_keys']
        dataset_params = self.hparams['dynamics_dataset_hparams']
        self.original_obs_feat_desc: Dict = dataset_params['observation_features_description']
        self.observation_features_description = {k: self.original_obs_feat_desc[k] for k in self.observation_features_keys}
        final_dim = sum(list(self.observation_features_description.values()))

        my_layers = []
        for h in self.hparams['observer_fc_layer_sizes']:
            my_layers.append(layers.Dense(h, activation="relu"))
        my_layers.append(layers.Dense(final_dim, activation=None))

        self.model = Sequential(my_layers)

    # @tf.function
    def call(self, inputs, **kwargs):
        z = tf.concat([inputs[k] for k in self.state_keys], axis=-1)
        y = self.model(z)
        y_dict = vector_to_dict(self.observation_features_description, y)
        return y_dict

    def compute_loss(self, dataset_element, outputs):
        raise NotImplementedError()


class CFMFilter(BaseFilterFunction):

    def make_net_and_checkpoint(self, batch_size, scenario):
        cfm = CFM(hparams=self.hparams, batch_size=batch_size, scenario=scenario)
        ckpt = tf.train.Checkpoint(model=cfm)
        return cfm.encoder, ckpt


class CFMLatentDynamics(BaseDynamicsFunction):

    def make_net_and_checkpoint(self, batch_size, scenario):
        cfm = CFM(hparams=self.hparams, batch_size=batch_size, scenario=scenario)
        ckpt = tf.train.Checkpoint(model=cfm)
        return cfm.dynamics, ckpt


def show_image_grid(images):
    import matplotlib.pyplot as plt
    from grid_strategy import strategies
    batch_size = images.shape[0]
    specs = strategies.SquareStrategy("center").get_grid(batch_size)
    for b, subplot in enumerate(specs):
        plt.subplot(subplot)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        if b == 0:
            ax.set_title(r"$I_t$")
            ax.imshow(images[b, 0, :, :, :3])
        elif b == 1:
            ax.set_title(r"$I_{t+1}$")
            ax.imshow(images[b - 1, 1, :, :, :3])
        else:
            ax.imshow(images[b - 1, 1, :, :, :3])
    plt.show()
