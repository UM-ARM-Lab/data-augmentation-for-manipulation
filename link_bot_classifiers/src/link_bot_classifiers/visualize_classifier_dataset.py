from time import perf_counter

import numpy as np
from colorama import Style
from matplotlib import pyplot as plt
from progressbar import progressbar
from scipy import stats

from link_bot_data import base_dataset
from link_bot_data.dataset_utils import add_predicted
from link_bot_pycommon.pycommon import print_dict
from moonshine.moonshine_utils import remove_batch


def visualize_dataset(args, classifier_dataset):
    tf_dataset = classifier_dataset.get_datasets(mode=args.mode, take=args.take)

    tf_dataset = tf_dataset.batch(1)

    iterator = iter(tf_dataset)
    t0 = perf_counter()

    reconverging_count = 0
    positive_count = 0
    negative_count = 0
    count = 0

    stdevs = []
    labels = []
    stdevs_for_negative = []
    stdevs_for_positive = []

    for i, example in enumerate(progressbar(tf_dataset, widgets=base_dataset.widgets)):
        if i < args.start_at:
            continue

        example = remove_batch(example)

        is_close = example['is_close'].numpy().squeeze()
        count += is_close.shape[0]

        n_close = np.count_nonzero(is_close[-1])
        n_far = is_close.shape[0] - n_close
        positive_count += n_close
        negative_count += n_far
        reconverging = n_far > 0 and is_close[-1]

        if args.only_reconverging and not reconverging:
            continue

        if args.only_negative and np.any(is_close[1:]):
            continue

        if args.only_positive and not np.any(is_close[1:]):
            continue

        # print(f"Example {i}, Trajectory #{int(example['traj_idx'])}")

        if count == 0:
            print_dict(example)

        if reconverging:
            reconverging_count += 1

        # Print statistics intermittently
        if count % 1000 == 0:
            print_stats_and_timing(args, count, reconverging_count, negative_count, positive_count)

        #############################
        # Show Visualization
        #############################
        if args.display_type == 'just_count':
            continue
        elif args.display_type == '3d':
            # print(example['is_close'])
            if example['is_close'][0] == 0:
                continue
            classifier_dataset.anim_transition_rviz(example)

        elif args.display_type == 'stdev':
            for t in range(1, classifier_dataset.horizon):
                stdev_t = example[add_predicted('stdev')][t, 0].numpy()
                label_t = example['is_close'][t]
                stdevs.append(stdev_t)
                labels.append(label_t)
                if label_t > 0.5:
                    stdevs_for_positive.append(stdev_t)
                else:
                    stdevs_for_negative.append(stdev_t)
        else:
            raise NotImplementedError()
    total_dt = perf_counter() - t0

    if args.display_type == 'stdev':
        print(f"p={stats.f_oneway(stdevs_for_negative, stdevs_for_positive)[1]}")

        plt.figure()
        plt.title(" ".join([str(d.name) for d in args.dataset_dirs]))
        bins = plt.hist(stdevs_for_negative, label='negative examples', alpha=0.8, density=True)[1]
        plt.hist(stdevs_for_positive, label='positive examples', alpha=0.8, bins=bins, density=True)
        plt.ylabel("count")
        plt.xlabel("stdev")
        plt.legend()
        plt.show()

    print_stats_and_timing(args, count, reconverging_count, negative_count, positive_count, total_dt)


def print_stats_and_timing(args, count, reconverging_count, negative_count, positive_count, total_dt=None):
    if args.perf and total_dt is not None:
        print("Total iteration time = {:.4f}".format(total_dt))
    class_balance = positive_count / count * 100
    print("Number of examples: {}".format(count))
    print("Number of reconverging examples: {}".format(reconverging_count))
    print("Number positive: {}".format(positive_count))
    print("Number negative: {}".format(negative_count))
    print("Class balance: {:4.1f}% positive".format(class_balance))


def compare_examples_from_datasets(args, classifier_dataset1, classifier_dataset2):
    tf_dataset1 = classifier_dataset1.get_datasets(mode=args.mode, take=args.take)
    tf_dataset2 = classifier_dataset2.get_datasets(mode=args.mode, take=args.take)

    tf_dataset1 = tf_dataset1.batch(1)
    tf_dataset2 = tf_dataset2.batch(1)

    datasets = tf_dataset1.zip(tf_dataset2)
    for i, (example1, example2) in enumerate(progressbar(datasets, widgets=base_dataset.widgets)):
        print(i, args.example_indices)

        example1 = remove_batch(example1)
        example2 = remove_batch(example2)

        is_close1 = example1['is_close'].numpy().squeeze()[1]
        is_close2 = example2['is_close'].numpy().squeeze()[1]

        if args.example_indices is not None and i not in args.example_indices:
            continue
        elif not args.show_all and not is_close1 == is_close2:
            continue

        status = f"Example {i}: " \
                 f"dataset {classifier_dataset1.name} has label {Style.BRIGHT}{is_close1}{Style.NORMAL}, " \
                 f"dataset {classifier_dataset2.name} has label {Style.BRIGHT}{is_close2}{Style.NORMAL}"
        print()
        print(status)
        print(f"Dataset 1, Example {i}")
        classifier_dataset1.anim_transition_rviz(example1)
        print(f"Dataset 2, Example {i}")
        classifier_dataset2.anim_transition_rviz(example2)
