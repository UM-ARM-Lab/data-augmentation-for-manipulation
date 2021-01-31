import datetime
import pathlib
import time
from typing import Optional

import progressbar
import tensorflow as tf
from colorama import Fore, Style

from moonshine.metric import LossMetric
from shape_completion_training.model.ae_vcnn import AE_VCNN
from shape_completion_training.model.augmented_ae import Augmented_VAE
from shape_completion_training.model.auto_encoder import AutoEncoder
from shape_completion_training.model.conditional_vcnn import ConditionalVCNN
from shape_completion_training.model.utils import (
    reduce_mean_dict, sequence_of_dicts_to_dict_of_sequences)
from shape_completion_training.model.vae import VAE, VAE_GAN
from shape_completion_training.model.voxelcnn import VoxelCNN


def get_model_type(network_type):
    if network_type == 'VoxelCNN':
        return VoxelCNN
    elif network_type == 'AutoEncoder':
        return AutoEncoder
    elif network_type == 'VAE':
        return VAE
    elif network_type == 'VAE_GAN':
        return VAE_GAN
    elif network_type == 'Augmented_VAE':
        return Augmented_VAE
    elif network_type == 'Conditional_VCNN':
        return ConditionalVCNN
    elif network_type == 'AE_VCNN':
        return AE_VCNN
    else:
        raise Exception('Unknown Model Type')


class ModelRunner:
    def __init__(self,
                 model,
                 training,
                 trial_path,
                 params,
                 checkpoint: Optional[pathlib.Path] = None,
                 key_metric=LossMetric,
                 val_every_n_batches=None,
                 mid_epoch_val_batches=None,
                 save_every_n_minutes: int = 60,
                 validate_first=False,
                 batch_metadata=None,
                 ):
        self.model = model
        self.training = training
        self.key_metric = key_metric
        self.trial_path = trial_path
        self.checkpoint = checkpoint
        self.params = params
        self.val_every_n_batches = val_every_n_batches
        self.mid_epoch_val_batches = mid_epoch_val_batches
        self.save_every_n_minutes = save_every_n_minutes
        self.overall_job_start_time = time.time()
        self.latest_minute = 0
        self.validate_first = validate_first
        if batch_metadata is None:
            self.batch_metadata = {}
        else:
            self.batch_metadata = batch_metadata

        self.group_name = self.trial_path.parts[-2]

        self.val_summary_writer = tf.summary.create_file_writer((self.trial_path / "logs/1_val").as_posix())
        self.train_summary_writer = tf.summary.create_file_writer((self.trial_path / "logs/2_train").as_posix())

        self.num_train_batches = None
        self.num_val_batches = None

        self.latest_ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                               epoch=tf.Variable(0),
                                               train_time=tf.Variable(0.0),
                                               best_key_metric_value=tf.Variable(self.key_metric.worst(), dtype=tf.float32),
                                               model=self.model)
        self.best_ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                             epoch=tf.Variable(0),
                                             train_time=tf.Variable(0.0),
                                             best_key_metric_value=tf.Variable(self.key_metric.worst(), dtype=tf.float32),
                                             model=self.model)

        self.latest_checkpoint_path = self.trial_path / "latest_checkpoint"
        self.best_checkpoint_path = self.trial_path / "best_checkpoint"
        self.latest_checkpoint_manager = tf.train.CheckpointManager(self.latest_ckpt,
                                                                    self.latest_checkpoint_path.as_posix(),
                                                                    max_to_keep=1)
        self.best_checkpoint_manager = tf.train.CheckpointManager(self.best_ckpt,
                                                                  self.best_checkpoint_path.as_posix(),
                                                                  max_to_keep=1)

        if self.checkpoint is not None:
            self.restore()

    def restore(self):
        restore_latest_checkpoint_path = self.checkpoint.parent / "latest_checkpoint"
        restore_best_checkpoint_path = self.checkpoint.parent / "best_checkpoint"
        restore_latest_checkpoint_manager = tf.train.CheckpointManager(self.latest_ckpt,
                                                                       restore_latest_checkpoint_path.as_posix(),
                                                                       max_to_keep=1)
        restore_best_checkpoint_manager = tf.train.CheckpointManager(self.best_ckpt,
                                                                     restore_best_checkpoint_path.as_posix(),
                                                                     max_to_keep=1)
        self.best_ckpt.restore(restore_best_checkpoint_manager.latest_checkpoint)
        if self.checkpoint.name == 'latest_checkpoint':
            status = self.latest_ckpt.restore(restore_latest_checkpoint_manager.latest_checkpoint)
            if restore_latest_checkpoint_manager.latest_checkpoint is not None:
                print(Fore.CYAN + "Restoring latest {}".format(restore_latest_checkpoint_manager.latest_checkpoint))
                status.assert_existing_objects_matched()
            else:
                raise ValueError("Failed to restore! wrong checkpoint path?")

    def count_params(self):
        self.model.summary()

    def build_model(self, dataset):
        elem = next(iter(dataset))
        tf.summary.trace_on(graph=True, profiler=False)
        self.model(elem, training=True)
        with self.train_summary_writer.as_default():
            tf.summary.trace_export(name='train_trace', step=self.latest_ckpt.step.numpy())

        model_image_path = self.trial_path / 'network.png'
        tf.keras.utils.plot_model(self.model, model_image_path.as_posix(), show_shapes=True)

    def write_individual_summary(self, k, v):
        if v.ndim == 0:
            tf.summary.scalar(k, v, step=self.latest_ckpt.step.numpy())
        elif v.ndim == 4:
            tf.summary.image(k, v, step=self.latest_ckpt.step.numpy())
        else:
            raise NotImplementedError(f"invalid number of dimensions in summary {v.ndim}")
        # TODO: gif summary?
        # if v.ndim == 5:
        #     tf.summary.video_scalar(k, v, step = self.latest_ckpt.step.numpy())

    def write_summary(self, writer, summary_dict):
        with writer.as_default():
            for k in summary_dict:
                v = summary_dict[k].numpy()
                self.write_individual_summary(k, v)

    def write_train_summary(self, summary_dict):
        self.write_summary(self.train_summary_writer, summary_dict)

    def write_val_summary(self, summary_dict):
        self.write_summary(self.val_summary_writer, summary_dict)

    def train_epoch(self, train_dataset, val_dataset):
        if self.num_train_batches is not None:
            max_size = str(self.num_train_batches)
        else:
            max_size = '???'

        widgets = [
            ' TRAIN ', progressbar.Counter(), '/', max_size,
            ' ', progressbar.Variable("Loss"), ' ',
            progressbar.Bar(),
            ' [', progressbar.Variable("TrainTime"), '] ',
            ' (', progressbar.ETA(), ') ',
        ]

        with progressbar.ProgressBar(widgets=widgets, max_value=self.num_train_batches) as bar:
            self.num_train_batches = 0
            t0 = time.time()

            for batch_idx, train_batch in enumerate(train_dataset):
                train_batch.update(self.batch_metadata)
                self.num_train_batches += 1
                self.latest_ckpt.step.assign_add(1)

                _, train_batch_metrics = self.model.train_step(train_batch)
                time_str = str(datetime.timedelta(seconds=int(self.latest_ckpt.train_time.numpy())))
                bar.update(self.num_train_batches,
                           Loss=train_batch_metrics['loss'].numpy().squeeze(), TrainTime=time_str)
                self.write_train_summary(train_batch_metrics)

                # Measure training time
                now = time.time()
                train_time = now - t0
                t0 = now
                self.latest_ckpt.train_time.assign_add(train_time)

                # Mid-epoch validation
                if self.val_every_n_batches is not None \
                        and batch_idx % self.val_every_n_batches == 0 \
                        and batch_idx > 0:
                    self.mid_epoch_validation(val_dataset)

                # Mid-epoch checkpointing
                overall_job_dt = now - self.overall_job_start_time
                current_minute = int(overall_job_dt // 60)
                if self.save_every_n_minutes \
                        and current_minute > self.latest_minute \
                        and current_minute % self.save_every_n_minutes == 0:
                    self.latest_minute = current_minute
                    save_path = self.latest_checkpoint_manager.save()
                    print("Saving " + save_path)

    def mid_epoch_validation(self, val_dataset):
        val_metrics = []
        for i, val_batch in enumerate(val_dataset.take(self.mid_epoch_val_batches)):
            val_batch.update(self.batch_metadata)
            _, val_batch_metrics = self.model.val_step(val_batch)
            val_metrics.append(val_batch_metrics)

        val_metrics = sequence_of_dicts_to_dict_of_sequences(val_metrics)
        mean_val_metrics = reduce_mean_dict(val_metrics)
        self.write_val_summary(mean_val_metrics)
        self.latest_checkpoint_manager.save()
        return mean_val_metrics

    def val_epoch(self, val_dataset):
        if self.num_val_batches is not None:
            max_size = str(self.num_val_batches)
        else:
            max_size = '???'

        widgets = [
            ' VAL   ', progressbar.Counter(), '/', max_size,
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
        ]

        with progressbar.ProgressBar(widgets=widgets, max_value=self.num_val_batches) as bar:
            self.num_val_batches = 0
            val_metrics = []
            for val_batch in val_dataset:
                val_batch.update(self.batch_metadata)
                self.num_val_batches += 1
                _, val_batch_metrics = self.model.val_step(val_batch)
                val_metrics.append(val_batch_metrics)
                bar.update(self.num_val_batches)

        val_metrics = sequence_of_dicts_to_dict_of_sequences(val_metrics)
        # TODO: we could get rid of this reduce mean if we used keras metrics properly... not all metrics should be averaged.
        mean_val_metrics = reduce_mean_dict(val_metrics)
        return mean_val_metrics

    def train(self, train_dataset, val_dataset, num_epochs):
        last_epoch = self.latest_ckpt.epoch + num_epochs
        try:
            # Validation before anything
            if self.validate_first:
                validation_metrics = self.val_epoch(val_dataset)
                self.write_val_summary(validation_metrics)
                key_metric_value = validation_metrics[self.key_metric.key()]
                print(Style.BRIGHT + "Val: {}={}".format(self.key_metric.key(), key_metric_value) + Style.NORMAL)

            while self.latest_ckpt.epoch < last_epoch:
                # Training
                self.latest_ckpt.epoch.assign_add(1)
                print('')
                msg_fmt = Fore.GREEN + Style.BRIGHT + 'Epoch {:3d}/{}, Group Name [{}]' + Style.RESET_ALL
                print(msg_fmt.format(self.latest_ckpt.epoch.numpy(), last_epoch, self.group_name))
                self.train_epoch(train_dataset, val_dataset)
                self.latest_checkpoint_manager.save()
                save_path = self.latest_checkpoint_manager.save()
                print(Fore.CYAN + "Saving " + save_path + Fore.RESET)

                # Validation at end of epoch
                validation_metrics = self.val_epoch(val_dataset)
                self.write_val_summary(validation_metrics)
                key_metric_value = validation_metrics[self.key_metric.key()]
                print(Style.BRIGHT + "Val: {}={}".format(self.key_metric.key(), key_metric_value) + Style.NORMAL)
                if self.key_metric.is_better_than(key_metric_value, self.best_ckpt.best_key_metric_value):
                    self.best_ckpt.best_key_metric_value.assign(key_metric_value)
                    self.latest_ckpt.best_key_metric_value.assign(key_metric_value)
                    save_path = self.best_checkpoint_manager.save()
                    print(Fore.CYAN + "New best checkpoint {}".format(save_path) + Fore.RESET)

        except KeyboardInterrupt:
            print(Fore.YELLOW + "Interrupted." + Fore.RESET)

        return validation_metrics
