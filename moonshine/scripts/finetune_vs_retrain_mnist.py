#!/usr/bin/env python

import logging
import pathlib
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from link_bot_pycommon.pkl_df_job_chunker import DfJobChunker

min_examples = 100
max_examples = 60_000
m_steps = 20
n_epochs = 10
n_seeds = 3


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def train_mnist(n_take: int, prefix: str, seed: int, checkpoint: Optional[str] = None):
    tf.random.set_seed(seed)

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
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples, seed=seed)
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
        model = tf.keras.models.load_model(checkpoint)

    history = model.fit(
        ds_train,
        epochs=n_epochs,
        validation_data=ds_test,
        verbose=False,
    )

    new_checkpoint = f'results/{prefix}_{n_take}'
    print(f'{new_checkpoint} {n_take} {seed}')
    model.save(new_checkpoint)

    final_validation_accuracy = history.history['val_sparse_categorical_accuracy'][-1]
    return new_checkpoint, final_validation_accuracy


def fine_tune(job_chunker):
    prefix = 'fine_tune'
    for seed in range(n_seeds):
        checkpoint = None
        for n_take in np.linspace(min_examples, max_examples, m_steps):
            n_take = int(n_take)
            row = {'seed': seed, 'n_take': n_take, 'prefix': prefix}
            if not job_chunker.has(row):
                checkpoint, final_validation_accuracy = train_mnist(n_take,
                                                                    prefix=prefix,
                                                                    checkpoint=checkpoint,
                                                                    seed=seed)
                row['final_validation_accuracy'] = final_validation_accuracy
                job_chunker.append(row)


def retrain(job_chunker):
    prefix = 'retrain'
    for seed in range(n_seeds):
        for n_take in np.linspace(min_examples, max_examples, m_steps):
            n_take = int(n_take)
            row = {'seed': seed, 'n_take': n_take, 'prefix': prefix}
            if not job_chunker.has(row):
                _, final_validation_accuracy = train_mnist(n_take,
                                                           prefix=prefix,
                                                           checkpoint=None,
                                                           seed=seed)
                row['final_validation_accuracy'] = final_validation_accuracy
                job_chunker.append(row)


def plot_results(root, chunker):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig = plt.figure()
    ax = plt.gca()
    sns.lineplot(ax=ax,
                 data=chunker.df,
                 x='n_take',
                 y='final_validation_accuracy',
                 hue='prefix',
                 ci=100,
                 palette='colorblind',
                 )
    ax.set_xlabel("# training examples")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Retraining vs Fine-Tuning: Online MNIST Classification")

    fig.savefig((root / 'comparison.png').as_posix())

    plt.show()


def main():
    tf.get_logger().setLevel(logging.ERROR)

    root = pathlib.Path('results/finetune_vs_restrain_mnist')
    root.mkdir(exist_ok=True, parents=True)
    df_filename = root / 'df.pkl'
    chunker = DfJobChunker(df_filename)

    fine_tune(chunker)
    retrain(chunker)

    plot_results(root, chunker)


if __name__ == '__main__':
    main()
