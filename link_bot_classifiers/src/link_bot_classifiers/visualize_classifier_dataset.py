from time import perf_counter
from typing import Dict

from colorama import Style
from matplotlib import pyplot as plt, colors
from scipy import stats
from tqdm import tqdm

import rospy
from link_bot_data.dataset_utils import add_predicted
from link_bot_data.tf_dataset_utils import deserialize_scene_msg
from moonshine import grid_utils_tf
from link_bot_pycommon.grid_utils_np import environment_to_vg_msg
from link_bot_pycommon.matplotlib_utils import adjust_lightness
from link_bot_pycommon.pycommon import print_dict
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.torch_and_tf_utils import remove_batch
from std_msgs.msg import ColorRGBA


def visualize_dataset(args, dataset_loader):
    dataset = dataset_loader.get_datasets(mode=args.mode, shuffle=args.shuffle)

    print_dict(dataset.get_example(0))
    print('*' * 100)

    dataset = dataset.take(args.take)
    dataset = dataset.skip(args.skip)
    dataset = dataset.shard(args.shard)

    t0 = perf_counter()

    positive_count = 0
    negative_count = 0
    starts_far_count = 0
    count = 0

    s = dataset_loader.get_scenario()

    def _make_stats_dict():
        return {
            'count':            count,
            'negative_count':   negative_count,
            'positive_count':   positive_count,
            'starts_far_count': starts_far_count,
        }

    stdevs = []
    labels = []
    stdevs_for_negative = []
    stdevs_for_positive = []

    for i, example in enumerate(tqdm(dataset)):

        deserialize_scene_msg(example)

        is_close = example['is_close'].numpy().squeeze()

        starts_far = is_close[0] == 0
        positive = bool(is_close[1])
        negative = not positive

        if args.only_negative and not negative:
            continue

        if args.only_positive and not positive:
            continue

        if args.only_starts_far and not starts_far:
            continue

        count += 1

        if positive:
            positive_count += 1

        if negative:
            negative_count += 1

        if starts_far:
            starts_far_count += 1

        # Print statistics intermittently
        if count % 1000 == 0:
            stats_dict = _make_stats_dict()
            print_stats_and_timing(args, stats_dict)

        #############################
        # Show Visualization
        #############################
        if args.display_type == 'just_count':
            continue
        elif args.display_type == 'volume':
            positions = [
                example[add_predicted('rope')].reshape([-1, 25, 3])[0, 12],
                example[add_predicted('left_gripper')][0],
                example[add_predicted('right_gripper')][0],
            ]
            s.plot_points_rviz(positions, label=f"{i}")
        elif args.display_type == '3d':
            if 'augmented_from' in example:
                print(f"augmented from: {example['augmented_from']}")
            s.plot_traj_idx_rviz(i)
            dataset_loader.anim_transition_rviz(example)
        elif args.display_type == 'stdev':
            for t in range(1, dataset_loader.horizon):
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

    stats_dict = _make_stats_dict()
    print_stats_and_timing(args, stats_dict, total_dt)


def print_stats_and_timing(args, counts: Dict, total_dt=None):
    if args.perf and total_dt is not None:
        print("Total iteration time = {:.4f}".format(total_dt))

    for name, count in counts.items():
        percentage = count / counts['count'] * 100
        print(f"{name} {count} ({percentage:.1f}%)")


def compare_examples_from_datasets(args, classifier_dataset1, classifier_dataset2):
    tf_dataset1 = classifier_dataset1
    tf_dataset2 = classifier_dataset2

    tf_dataset1 = tf_dataset1.batch(1)
    tf_dataset2 = tf_dataset2.batch(1)

    datasets = tf_dataset1.zip(tf_dataset2)
    for i, (example1, example2) in enumerate(tqdm(datasets)):
        print(i, args.example_indices)

        example1 = remove_batch(example1)
        example2 = remove_batch(example2)

        is_close1 = example1['is_close'].numpy().squeeze()[1]
        is_close2 = example2['is_close'].numpy().squeeze()[1]

        if args.example_indices is not None and i not in args.example_indices:
            continue
        elif not args.SHOW_ALL and not is_close1 == is_close2:
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


def viz_compare_examples(s: ScenarioWithVisualization,
                         aug_example: Dict,
                         data_example: Dict,
                         aug_env_pub: rospy.Publisher,
                         data_env_pub: rospy.Publisher,
                         use_predicted: bool = False):
    if aug_example is not None:
        viz_compare_example(s, aug_example, 'aug', aug_env_pub, color='#aa2222', use_predicted=use_predicted)
    if data_example is not None:
        viz_compare_example(s, data_example, 'data', data_env_pub, color='#2222aa', use_predicted=use_predicted)


def viz_compare_example(s: ScenarioWithVisualization,
                        e: Dict,
                        label: str,
                        env_pub: rospy.Publisher,
                        color,
                        use_predicted: bool = False):
    state_before = {
        'rope':            e[add_predicted('rope') if use_predicted else 'rope'][0],
        'joint_positions': e['joint_positions'][0],
        'joint_names':     e['joint_names'][0],
    }
    state_after = {
        'rope':            e[add_predicted('rope') if use_predicted else 'rope'][1],
        'joint_positions': e['joint_positions'][1],
        'joint_names':     e['joint_names'][0],
    }
    s.plot_state_rviz(state_before, label=label + '_before', color=adjust_lightness(color, 0.6))
    s.plot_state_rviz(state_after, label=label + '_after', color=adjust_lightness(color, 0.8))
    env = {
        'env':          e['env'],
        'res':          e['res'],
        'origin_point': e['origin_point'],
        # 'res':          0.02,
        # 'origin_point': np.array([1.0, 0, 0]),
    }
    s.plot_environment_rviz(env)

    frame = 'env_vg'
    color_rgba = ColorRGBA(*colors.to_rgba(color))
    env_msg = environment_to_vg_msg(env, frame=frame, color=color_rgba)
    env_pub.publish(env_msg)
    grid_utils_tf.send_voxelgrid_tf_origin_point_res(s.tf.tf_broadcaster,
                                                     env['origin_point'],
                                                     env['res'],
                                                     child_frame_id=frame)
    s.tf.send_transform(env['origin_point'], [0, 0, 0, 1], 'world', child='origin_point')
    s.plot_is_close(e['is_close'][1])
