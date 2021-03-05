import datetime
import pathlib
import time
from typing import Optional

import progressbar
import tensorflow as tf
from colorama import Fore, Style

from moonshine.metrics import LossCheckpointMetric
from moonshine.my_keras_model import MyKerasModel


class ModelRunner:
    def __init__(self,
                 model: MyKerasModel,
                 training,
                 trial_path,
                 params,
                 checkpoint: Optional[pathlib.Path] = None,
                 key_metric=LossCheckpointMetric,
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
                                               best_key_metric_value=tf.Variable(self.key_metric.worst(),
                                                                                 dtype=tf.float32),
                                               model=self.model)
        self.best_ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                             epoch=tf.Variable(0),
                                             train_time=tf.Variable(0.0),
                                             best_key_metric_value=tf.Variable(self.key_metric.worst(),
                                                                               dtype=tf.float32),
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
        best_checkpoint_manager = self.get_checkpoint_manager('best_checkpoint', self.best_ckpt)
        latest_checkpoint_manager = self.get_checkpoint_manager('latest_checkpoint', self.latest_ckpt)
        self.best_ckpt.restore(best_checkpoint_manager.latest_checkpoint)
        if self.checkpoint.name == 'latest_checkpoint':
            status = self.latest_ckpt.restore(latest_checkpoint_manager.latest_checkpoint)
            if latest_checkpoint_manager.latest_checkpoint is not None:
                print(Fore.CYAN + "Restoring latest {}".format(latest_checkpoint_manager.latest_checkpoint))
                status.assert_existing_objects_matched()
            else:
                raise ValueError("Failed to restore! wrong checkpoint path?")

    def get_checkpoint_manager(self, name: str, ckpt: tf.train.Checkpoint):
        checkpoint_path = self.checkpoint.parent / name
        checkpoint_manager = tf.train.CheckpointManager(ckpt, checkpoint_path.as_posix(), max_to_keep=1)
        return checkpoint_manager

    def set_best_ckpt_step_from_latest_ckpt(self):
        latest_checkpoint_manager = self.get_checkpoint_manager('latest_checkpoint', self.latest_ckpt)
        self.latest_ckpt.restore(latest_checkpoint_manager.latest_checkpoint)
        if latest_checkpoint_manager.latest_checkpoint is not None:
            self.best_ckpt.step.assign(self.latest_ckpt.step)
        else:
            raise ValueError("Failed to restore! wrong checkpoint path?")

    def write_individual_summary(self, k, v):
        if v.ndim == 0:
            tf.summary.scalar(k, v, step=self.latest_ckpt.step.numpy())
        elif v.ndim == 1 and tf.size(v).numpy() == 1:
            tf.summary.scalar(k, v[0], step=self.latest_ckpt.step.numpy())
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

    def train_epoch(self, train_dataset, val_dataset, train_metrics, val_metrics):
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

                for v in train_metrics.values():
                    v.reset_states()
                _ = self.model.train_step(train_batch, train_metrics)
                time_str = str(datetime.timedelta(seconds=int(self.latest_ckpt.train_time.numpy())))
                train_batch_loss = train_metrics['loss'].result().numpy().squeeze()
                bar.update(self.num_train_batches, Loss=train_batch_loss, TrainTime=time_str)
                self.write_train_summary({k: m.result() for k, m in train_metrics.items()})

                # Measure training time
                now = time.time()
                train_time = now - t0
                t0 = now
                self.latest_ckpt.train_time.assign_add(train_time)

                # Mid-epoch validation
                if self.val_every_n_batches is not None \
                        and batch_idx % self.val_every_n_batches == 0 \
                        and batch_idx > 0:
                    self.mid_epoch_validation(val_dataset, val_metrics)

                # Mid-epoch checkpointing
                overall_job_dt = now - self.overall_job_start_time
                current_minute = int(overall_job_dt // 60)
                if self.save_every_n_minutes \
                        and current_minute > self.latest_minute \
                        and current_minute % self.save_every_n_minutes == 0:
                    self.latest_minute = current_minute
                    save_path = self.latest_checkpoint_manager.save()
                    print("Saving " + save_path)

    def mid_epoch_validation(self, val_dataset, val_metrics):
        for v in val_metrics.values():
            v.reset_states()

        for i, val_batch in enumerate(val_dataset.take(self.mid_epoch_val_batches)):
            val_batch.update(self.batch_metadata)
            _ = self.model.val_step(val_batch, val_metrics)

        self.write_val_summary({k: m.result() for k, m in val_metrics.items()})
        self.latest_checkpoint_manager.save()

    def val_epoch(self, val_dataset, val_metrics):
        if self.num_val_batches is not None:
            max_size = str(self.num_val_batches)
        else:
            max_size = '???'

        widgets = [
            ' VAL   ', progressbar.Counter(), '/', max_size,
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
        ]

        for v in val_metrics.values():
            v.reset_states()

        with progressbar.ProgressBar(widgets=widgets, max_value=self.num_val_batches) as bar:
            self.num_val_batches = 0
            for val_batch in val_dataset:
                val_batch.update(self.batch_metadata)
                self.num_val_batches += 1
                _ = self.model.val_step(val_batch, val_metrics)
                bar.update(self.num_val_batches)

    def train(self, train_dataset, val_dataset, num_epochs):
        val_metrics = self.model.create_metrics()
        train_metrics = self.model.create_metrics()

        last_epoch = self.latest_ckpt.epoch + num_epochs
        try:
            # Validation before anything
            if self.validate_first:
                self.val_epoch(val_dataset, val_metrics)
                self.write_val_summary({k: m.result() for k, m in val_metrics.items()})
                key_metric_value = val_metrics[self.key_metric.key()].result()
                print(Style.BRIGHT + "Val: {}={}".format(self.key_metric.key(), key_metric_value) + Style.NORMAL)

            while self.latest_ckpt.epoch < last_epoch:
                # Training
                self.latest_ckpt.epoch.assign_add(1)
                print('')
                msg_fmt = Fore.GREEN + Style.BRIGHT + 'Epoch {:3d}/{}, Group Name [{}]' + Style.RESET_ALL
                print(msg_fmt.format(self.latest_ckpt.epoch.numpy(), last_epoch, self.group_name))
                self.train_epoch(train_dataset, val_dataset, train_metrics, val_metrics)
                self.latest_checkpoint_manager.save()
                save_path = self.latest_checkpoint_manager.save()
                print(Fore.CYAN + "Saving " + save_path + Fore.RESET)

                # Validation at end of epoch
                self.val_epoch(val_dataset, val_metrics)
                self.write_val_summary({k: m.result() for k, m in val_metrics.items()})
                key_metric_value = val_metrics[self.key_metric.key()].result()
                print(Style.BRIGHT + "Val: {}={}".format(self.key_metric.key(), key_metric_value) + Style.NORMAL)
                if self.key_metric.is_better_than(key_metric_value, self.best_ckpt.best_key_metric_value):
                    self.best_ckpt.best_key_metric_value.assign(key_metric_value)
                    self.latest_ckpt.best_key_metric_value.assign(key_metric_value)
                    save_path = self.best_checkpoint_manager.save()
                    print(Fore.CYAN + "New best checkpoint {}".format(save_path) + Fore.RESET)

        except KeyboardInterrupt:
            print(Fore.YELLOW + "Interrupted." + Fore.RESET)

        return val_metrics
