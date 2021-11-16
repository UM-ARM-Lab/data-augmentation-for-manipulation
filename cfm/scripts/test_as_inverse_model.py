import argparse
import pathlib

import colorama
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import rospy
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_planning.my_planner import PlanningQuery
from link_bot_planning.shooting_method import ShootingMethod
from link_bot_planning.trajectory_optimizer import TrajectoryOptimizer
from link_bot_pycommon.ros_pycommon import publish_color_image
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.moonshine_utils import remove_batch, add_batch
from moonshine.numpify import numpify
from sensor_msgs.msg import Image
from state_space_dynamics import dynamics_utils, filter_utils


# limit_gpu_mem(10)


def main():
    tf.random.set_seed(0)
    np.random.seed(0)
    colorama.init(autoreset=True)
    np.set_printoptions(linewidth=200, precision=3, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dirs", type=pathlib.Path, nargs="+")
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("--mode", type=str, choices=['train', 'val', 'test'], default='val')

    args = parser.parse_args()

    # TODO: REMOVE ME!
    args.mode = 'train'

    rospy.init_node("test_as_inverse_model")

    test_dataset = DynamicsDatasetLoader(args.dataset_dirs)
    test_tf_dataset = test_dataset.get_datasets(mode=args.mode)

    filter_model = filter_utils.load_filter([args.checkpoint])
    latent_dynamics_model = dynamics_utils.load_generic_model(args.checkpoint)

    test_as_inverse_model(filter_model, latent_dynamics_model, test_dataset, test_tf_dataset)


def test_as_inverse_model(filter_model, latent_dynamics_model, test_dataset, test_tf_dataset):
    scenario = test_dataset.scenario
    shooting_method = ShootingMethod(fwd_model=latent_dynamics_model,
                                     classifier_model=None,
                                     scenario=scenario,
                                     params={
                                         'n_samples': 1000
                                     })
    trajopt = TrajectoryOptimizer(fwd_model=latent_dynamics_model,
                                  classifier_model=None,
                                  scenario=scenario,
                                  params={
                                      "iters":                 100,
                                      "length_alpha":          0,
                                      "goal_alpha":            1000,
                                      "constraints_alpha":     0,
                                      "action_alpha":          0,
                                      "initial_learning_rate": 0.0001,
                                  }) # FIXME: custom cost function?

    s_color_viz_pub = rospy.Publisher("s_state_color_viz", Image, queue_size=10, latch=True)
    s_next_color_viz_pub = rospy.Publisher("s_next_state_color_viz", Image, queue_size=10, latch=True)
    image_diff_viz_pub = rospy.Publisher("image_diff_viz", Image, queue_size=10, latch=True)

    action_horizon = 1
    initial_actions = []
    total_errors = []
    for example_idx, example in enumerate(test_tf_dataset):
        stepper = RvizAnimationController(n_time_steps=test_dataset.steps_per_traj)
        for t in range(test_dataset.steps_per_traj - 1):
            print(example_idx)
            environment = {}
            current_observation = remove_batch(scenario.index_observation_time_batched(add_batch(example), t))

            for j in range(action_horizon):
                left_gripper_position = [0, 0, 0]
                right_gripper_position = [0, 0, 0]
                initial_action = {
                    'left_gripper_position':  left_gripper_position,
                    'right_gripper_position': right_gripper_position,
                }
                initial_actions.append(initial_action)
            goal_observation = {k: example[k][1] for k in filter_model.obs_keys}
            planning_query = PlanningQuery(start=current_observation, goal=goal_observation, environment=environment, seed=1)
            planning_result = shooting_method.plan(planning_query)
            actions = planning_result.actions
            planned_path = planning_result.latent_path
            true_action = numpify({k: example[k][0] for k in latent_dynamics_model.action_keys})

            for j in range(action_horizon):
                optimized_action = actions[j]
                # optimized_action = {
                #     'left_gripper_position': current_observation['left_gripper'],
                #     'right_gripper_position': current_observation['right_gripper'],
                # }
                true_action = numpify({k: example[k][j] for k in latent_dynamics_model.action_keys})

                # Visualize
                s = numpify(remove_batch(scenario.index_observation_time_batched(add_batch(example), 0)))
                s.update(numpify(remove_batch(scenario.index_observation_features_time_batched(add_batch(example), 0))))
                s_next = numpify(remove_batch(scenario.index_observation_time_batched(add_batch(example), 1)))
                s_next.update(numpify(remove_batch(scenario.index_observation_features_time_batched(add_batch(example), 1))))
                scenario.plot_state_rviz(s, label='t', color="#ff000055", id=1)
                scenario.plot_state_rviz(s_next, label='t+1', color="#aa222255", id=2)
                # scenario.plot_action_rviz(s, optimized_action, label='inferred', color='#00ff00', id=1)
                # scenario.plot_action_rviz(s, true_action, label='true', color='#ee770055', id=2)

                publish_color_image(s_color_viz_pub, s['rgbd'][:, :, :3])
                publish_color_image(s_next_color_viz_pub, s_next['rgbd'][:, :, :3])
                diff = s['rgbd'][:, :, :3] - s_next['rgbd'][:, :, :3]
                publish_color_image(image_diff_viz_pub, diff)

                # Metrics
                total_error = 0
                for v1, v2 in zip(optimized_action.values(), true_action.values()):
                    total_error += -np.dot(v1, v2)
                total_errors.append(total_error)

                stepper.step()

        if example_idx > 100:
            break
    print(np.min(total_errors))
    print(np.max(total_errors))
    print(np.mean(total_errors))
    plt.xlabel("total error (meter-ish)")
    plt.hist(total_errors, bins=np.linspace(0, 2, 20))
    plt.show()


if __name__ == '__main__':
    main()
