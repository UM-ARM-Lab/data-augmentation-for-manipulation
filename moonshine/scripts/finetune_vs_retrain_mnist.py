#!/usr/bin/env python
import logging
import pathlib
from typing import Optional

import tensorflow as tf
import tensorflow_datasets as tfds

from link_bot_pycommon.job_chunking import JobChunker

resolution = 100_000
max_examples = 1_000_000
n_epochs = 10


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def train_mnist(n_take: int, prefix: str, checkpoint: Optional[str] = None):
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    ds_train = ds_train.map(normalize_img)

    # here we take a subset of the training, dataset
    ds_train = ds_train.take(n_take)

    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(normalize_img)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    if checkpoint is not None:
        print(f'restoring {checkpoint}')
        model = tf.keras.models.load_model(checkpoint)

    history = model.fit(
        ds_train,
        epochs=n_epochs,
        validation_data=ds_test,
        verbose=False,
    )

    new_checkpoint = f'results/{prefix}_{n_take}'
    model.save(new_checkpoint)

    final_validation_accuracy = history.history['val_sparse_categorical_accuracy'][-1]
    return new_checkpoint, final_validation_accuracy


def fine_tune(job_chunker):
    checkpoint = None
    for n_take in range(resolution, max_examples, resolution):
        if job_chunker.get_result(str(n_take)) is None:
            checkpoint, final_validation_accuracy = train_mnist(n_take, 'fine_tune', checkpoint=checkpoint)
            job_chunker.store_result(str(n_take), final_validation_accuracy)


def retrain(job_chunker):
    for n_take in range(resolution, max_examples, resolution):
        if job_chunker.get_result(str(n_take)) is None:
            _, final_validation_accuracy = train_mnist(n_take, 'retrain', checkpoint=None)
            job_chunker.store_result(str(n_take), final_validation_accuracy)


def plot_results(root, finetune_chunker, retrain_chunker):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.gca()
    x = [int(i) for i in finetune_chunker.log.keys()]
    ft_y = list(finetune_chunker.log.values())
    rt_y = list(retrain_chunker.log.values())
    ax.plot(x, ft_y, label='fine tune')
    ax.plot(x, rt_y, label='retrain')
    ax.set_xlabel("# training examples")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Retraining vs Fine-Tuning: Online MNIST Classification")
    ax.legend()

    fig.savefig((root / 'comparison.png').as_posix())

    plt.show()


def main():
    tf.get_logger().setLevel(logging.ERROR)

    root = pathlib.Path('results/finetune_vs_restrain_mnist')
    root.mkdir(exist_ok=True, parents=True)
    job_chunker = JobChunker(root / 'logfile.hjson')
    finetune_chunker = job_chunker.sub_chunker('fine_tune')
    retrain_chunker = job_chunker.sub_chunker('retrain')

    fine_tune(finetune_chunker)
    retrain(retrain_chunker)

    plot_results(root, finetune_chunker, retrain_chunker)


if __name__ == '__main__':
    main()
