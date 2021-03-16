import logging
import pathlib
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold, cluster, decomposition
import tensorflow as tf

from arc_utilities.ros_init import rospy_and_cpp_init
from link_bot_classifiers.classifier_utils import load_generic_model, make_max_class_prob
from link_bot_classifiers.nn_classifier import NNClassifierWrapper
from link_bot_classifiers.train_test_classifier import filter_dataset_with_classifier
from link_bot_data.visualization import noisey_1d_scatter
from moonshine.ensemble import Ensemble2

rospy_and_cpp_init('compare_fp_tn_ensemble')

tf.get_logger().setLevel(logging.ERROR)


def compute_decisions_and_labels(example: Dict, predictions: Dict):
    probabilities = predictions['probabilities']
    decisions = tf.squeeze(probabilities > 0.5, axis=-1)
    labels = tf.expand_dims(example['is_close'][:, 1:], axis=2)
    labels = tf.squeeze(tf.cast(labels, tf.bool), axis=-1)

    return decisions, labels


def keep_tn(example: Dict, predictions: Dict):
    decisions, labels = compute_decisions_and_labels(example, predictions)

    is_tn = tf.logical_and(tf.logical_not(labels), tf.logical_not(decisions))

    return tf.squeeze(is_tn)


def keep_fp(example: Dict, predictions: Dict):
    decisions, labels = compute_decisions_and_labels(example, predictions)

    # classifier_is_correct = tf.equal(decisions, labels)
    # is_tp = tf.logical_and(labels, decisions)
    # is_tn = tf.logical_and(tf.logical_not(labels), tf.logical_not(decisions))
    is_fp = tf.logical_and(tf.logical_not(labels), decisions)

    return tf.squeeze(is_fp)


def generate_and_save(outfile):
    take = 500
    val_car_dir = [pathlib.Path('classifier_data/val_car_feasible_1614981888')]
    checkpoint = pathlib.Path('trials/val_car_feasible_1614981888/March_05_18-57-54_4b65490ac1/best_checkpoint')
    car_tn_itr = filter_dataset_with_classifier(dataset_dirs=val_car_dir,
                                                checkpoint=checkpoint,
                                                mode='val',
                                                use_gt_rope=True,
                                                should_keep_example=keep_tn,
                                                take_after_filter=take,
                                                )
    dataset, model = next(car_tn_itr)

    rack_dir = [pathlib.Path('classifier_data/results/rack1')]
    rack_fp_itr = filter_dataset_with_classifier(dataset_dirs=val_car_dir,
                                                 checkpoint=checkpoint,
                                                 mode='val',
                                                 use_gt_rope=True,
                                                 should_keep_example=keep_fp,
                                                 take_after_filter=take,
                                                 )

    next(rack_fp_itr)

    checkpoints = [
        pathlib.Path('trials/val_car_feasible_ensemble/0/best_checkpoint'),
        pathlib.Path('trials/val_car_feasible_ensemble/1/best_checkpoint'),
        pathlib.Path('trials/val_car_feasible_ensemble/2/best_checkpoint'),
        pathlib.Path('trials/val_car_feasible_ensemble/3/best_checkpoint'),
    ]
    models = [load_generic_model(checkpoint) for checkpoint in checkpoints]
    ensemble = Ensemble2(models, [])

    rack_fp_probabilities = []
    rack_fp_stdevs = []
    rack_fp_hs = []
    for batch_idx, example, _ in rack_fp_itr:
        mean_ensemble_predictions, stdev_ensemble_predictions = ensemble(NNClassifierWrapper.check_constraint_from_example,
                                                                         example)
        p = mean_ensemble_predictions['probabilities'].numpy().squeeze()
        s = stdev_ensemble_predictions['probabilities'].numpy().squeeze()
        h = mean_ensemble_predictions['out_h'].numpy().squeeze().flatten()
        rack_fp_probabilities.append(p)
        rack_fp_stdevs.append(s)
        rack_fp_hs.append(h)

    car_tn_probabilities = []
    car_tn_stdevs = []
    car_tn_hs = []
    for batch_idx, example, _ in car_tn_itr:
        mean_ensemble_predictions, stdev_ensemble_predictions = ensemble(NNClassifierWrapper.check_constraint_from_example,
                                                                         example)
        p = mean_ensemble_predictions['probabilities'].numpy().squeeze()
        s = stdev_ensemble_predictions['probabilities'].numpy().squeeze()
        h = mean_ensemble_predictions['out_h'].numpy().squeeze().flatten()
        car_tn_probabilities.append(p)
        car_tn_stdevs.append(s)
        car_tn_hs.append(h)

    data = {
        'car_tn_probabilities':  np.array(car_tn_probabilities),
        'car_tn_stdevs':         np.array(car_tn_stdevs),
        'car_tn_hs':             np.array(car_tn_hs),
        'rack_fp_probabilities': np.array(rack_fp_probabilities),
        'rack_fp_stdevs':        np.array(rack_fp_stdevs),
        'rack_fp_hs':            np.array(rack_fp_hs),
    }

    print(outfile.as_posix())
    np.savez(outfile, **data)

    return data


outfile = pathlib.Path("results/car_tn_vs_rack_fp.npz")
data = generate_and_save(outfile)

data = np.load(outfile.as_posix())

car_tn_probabilities = data['car_tn_probabilities']
car_tn_mcp = make_max_class_prob(car_tn_probabilities)
car_tn_stdevs = data['car_tn_stdevs']
car_tn_hs = data['car_tn_hs']

rack_fp_probabilities = data['rack_fp_probabilities']
rack_fp_mcp = make_max_class_prob(rack_fp_probabilities)
rack_fp_stdevs = data['rack_fp_stdevs']
rack_fp_hs = data['rack_fp_hs']

pca = decomposition.PCA(n_components=3)
car_tn_hs_2d = pca.fit_transform(car_tn_hs)
rack_fp_hs_2d = pca.transform(rack_fp_hs)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*car_tn_hs_2d.T, label='car TN')
ax.scatter(*rack_fp_hs_2d.T, label='rack FP')
ax.set_title("clustering t-sne projection of the transition latent vector")
ax.legend()

_, ax = plt.subplots()
ax.violinplot([car_tn_mcp, rack_fp_mcp], [1, 2], widths=0.8)
ax.set_xticks([1, 2])
ax.set_xticklabels(['car TN', 'rack FP'])
ax.set_xlabel("density")
ax.set_ylabel("max class probability")
s = 50
lw = 0.3
noise = 0.05
noisey_1d_scatter(ax, car_tn_mcp, 1, noise=noise, edgecolors='m', s=s, facecolors='none', linewidth=lw)
noisey_1d_scatter(ax, rack_fp_mcp, 2, noise=noise, edgecolors='m', s=s, facecolors='none', linewidth=lw)
plt.show()

_, ax = plt.subplots()
ax.violinplot([car_tn_stdevs, rack_fp_stdevs], [1, 2], widths=0.8)
ax.set_xticks([1, 2])
ax.set_xticklabels(['car TN', 'rack FP'])
ax.set_xlabel("density")
ax.set_ylabel("ensemble stdev")
noisey_1d_scatter(ax, car_tn_stdevs, 1, noise=noise, edgecolors='m', s=s, facecolors='none', linewidth=lw)
noisey_1d_scatter(ax, rack_fp_stdevs, 2, noise=noise, edgecolors='m', s=s, facecolors='none', linewidth=lw)
plt.show()

pass