#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import rospy
import tensorflow
from colorama import Fore
from link_bot_gazebo.msg import LinkBotVelocityAction
from link_bot_gazebo.srv import WorldControlRequest, LinkBotStateRequest

from link_bot_data import random_environment_data_utils
from link_bot_data.link_bot_dataset_utils import float_feature
from link_bot_gazebo import gazebo_utils
from link_bot_planning.goals import sample_goal
from link_bot_planning.params import LocalEnvParams, FullEnvParams, SimParams
from link_bot_pycommon import ros_pycommon
from link_bot_pycommon.args import my_formatter

opts = tensorflow.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
conf = tensorflow.compat.v1.ConfigProto(gpu_options=opts)
tensorflow.compat.v1.enable_eager_execution(config=conf)


def generate_traj(args, services, traj_idx, global_t_step, gripper1_target_x, gripper1_target_y, goal_rng: np.random.RandomState):
    # FIXME: don't use separate XY
    gripper1_target = np.array([gripper1_target_x, gripper1_target_y])
    state_req = LinkBotStateRequest()
    action_msg = LinkBotVelocityAction()

    # At this point, we hope all of the objects have stopped moving, so we can get the environment and assume it never changes
    # over the course of this function
    full_env_data = gazebo_utils.get_occupancy_data(env_w=args.env_w, env_h=args.env_h, res=args.res, services=services)

    combined_constraint_labels = np.ndarray((args.steps_per_traj, 1))
    feature = {
        'local_env_rows': float_feature(np.array([args.local_env_rows])),
        'local_env_cols': float_feature(np.array([args.local_env_cols])),
        'full_env/env': float_feature(full_env_data.data.flatten()),
        'full_env/extent': float_feature(np.array(full_env_data.extent)),
        'full_env/origin': float_feature(full_env_data.origin),
    }

    for time_idx in range(args.steps_per_traj):
        # Query the current state
        state = services.get_state(state_req)
        head_idx = state.link_names.index("head")
        rope_configuration = gazebo_utils.points_to_config(state.points)
        head_point = state.points[head_idx]

        # Pick a new target if necessary
        if global_t_step % args.steps_per_target == 0:
            gripper1_target = sample_goal(args.env_w, args.env_h, head_point, env_padding=0.5, rng=goal_rng)
            gripper1_target_x, gripper1_target_y = gripper1_target
            if args.verbose:
                print('gripper target:', gripper1_target_x, gripper1_target_y)
                random_environment_data_utils.publish_marker(gripper1_target_x, gripper1_target_y, marker_size=0.05)

        # compute the velocity to move in that direction
        velocity = np.minimum(np.maximum(np.random.randn() * 0.07 + 0.10, 0), 0.15)
        dpos = gripper1_target - np.array([head_point.x, head_point.y])
        dpos_unit = dpos / np.linalg.norm(dpos)
        gripper1_target_v = velocity * dpos
        gripper1_target_vx = gripper1_target_v[0]
        gripper1_target_vy = gripper1_target_v[1]
        # gripper1_target_vx = velocity if gripper1_target_x > head_point.x else -velocity
        # gripper1_target_vy = velocity if gripper1_target_y > head_point.y else -velocity

        # publish the pull command, which will return the target velocity
        action_msg.gripper1_velocity.x = gripper1_target_vx
        action_msg.gripper1_velocity.y = gripper1_target_vy
        services.velocity_action_pub.publish(action_msg)

        # let the simulator run
        step = WorldControlRequest()
        step.steps = int(args.dt / args.max_step_size)  # assuming 0.001s per simulation step
        services.world_control(step)  # this will block until stepping is complete

        # NOTE: at_constraint_boundary is not used at the moment
        post_action_state = services.get_state(state_req)
        stopped = 0.025  # This threshold must be tuned whenever physics or the above velocity controller parameters change
        if (abs(post_action_state.gripper1_velocity.x) < stopped < abs(gripper1_target_vx)) \
                or (abs(post_action_state.gripper1_velocity.y) < stopped < abs(gripper1_target_vy)):
            at_constraint_boundary = True
        else:
            at_constraint_boundary = False

        if args.verbose:
            print("{} {:0.4f} {:0.4f} {:0.4f} {:0.4f} {}".format(time_idx,
                                                                 gripper1_target_vx,
                                                                 gripper1_target_vy,
                                                                 post_action_state.gripper1_velocity.x,
                                                                 post_action_state.gripper1_velocity.y,
                                                                 at_constraint_boundary))

        combined_constraint_labels[time_idx, 0] = at_constraint_boundary

        # format the tf feature
        head_np = np.array([head_point.x, head_point.y])
        local_env_data = gazebo_utils.get_local_occupancy_data(args.local_env_rows,
                                                               args.local_env_cols,
                                                               args.res,
                                                               center_point=head_np,
                                                               services=services)

        # for compatibility with video prediction
        feature['{}/endeffector_pos'.format(time_idx)] = float_feature(head_np)
        # for debugging/visualizing the constraint label
        feature['{}/1/velocity'.format(time_idx)] = float_feature(
            np.array([state.gripper1_velocity.x, state.gripper1_velocity.y]))
        feature['{}/1/post_action_velocity'.format(time_idx)] = float_feature(
            np.array([post_action_state.gripper1_velocity.x, post_action_state.gripper1_velocity.y]))
        feature['{}/1/force'.format(time_idx)] = float_feature(np.array([state.gripper1_force.x, state.gripper1_force.y]))
        # for learning dynamics in rope configuration space
        feature['{}/rope_configuration'.format(time_idx)] = float_feature(rope_configuration.flatten())
        feature['{}/state'.format(time_idx)] = float_feature(rope_configuration.flatten())
        feature['{}/action'.format(time_idx)] = float_feature(np.array([gripper1_target_vx, gripper1_target_vy]))
        feature['{}/constraint'.format(time_idx)] = float_feature(np.array([float(at_constraint_boundary)]))
        feature['{}/actual_local_env/env'.format(time_idx)] = float_feature(local_env_data.data.flatten())
        feature['{}/actual_local_env/extent'.format(time_idx)] = float_feature(np.array(local_env_data.extent))
        feature['{}/actual_local_env/origin'.format(time_idx)] = float_feature(local_env_data.origin)
        feature['{}/res'.format(time_idx)] = float_feature(np.array([local_env_data.resolution[0]]))
        feature['{}/traj_idx'.format(time_idx)] = float_feature(np.array([traj_idx]))
        feature['{}/time_idx'.format(time_idx)] = float_feature(np.array([time_idx]))

        global_t_step += 1

    n_positive = np.count_nonzero(np.any(combined_constraint_labels, axis=1))
    percentage_positive = n_positive * 100.0 / combined_constraint_labels.shape[0]

    if args.verbose:
        print(Fore.GREEN + "Trajectory {} Complete".format(traj_idx) + Fore.RESET)

    example_proto = tensorflow.train.Example(features=tensorflow.train.Features(feature=feature))
    # TODO: include documentation *inside* the tfrecords file describing what each feature is
    example = example_proto.SerializeToString()
    return example, percentage_positive, global_t_step, gripper1_target_x, gripper1_target_y


def generate_trajs(args, full_output_directory, services, gazebo_rng: np.random.RandomState, goal_rng: np.random.RandomState):
    examples = np.ndarray([args.trajs_per_file], dtype=object)
    percentages_positive = []
    global_t_step = 0
    gripper1_target_x = None
    gripper1_target_y = None
    for i in range(args.trajs):
        current_record_traj_idx = i % args.trajs_per_file

        if not args.no_obstacles and i % args.move_objects_every_n == 0:
            objects = ['moving_box{}'.format(i) for i in range(1, 7)]
            gazebo_utils.move_objects(services,
                                      args.max_step_size,
                                      objects,
                                      args.env_w,
                                      args.env_h,
                                      'velocity',
                                      padding=0.5,
                                      rng=gazebo_rng)

        # Generate a new trajectory
        example, percentage_violation, global_t_step, gripper1_target_x, gripper1_target_y = generate_traj(args, services, i,
                                                                                                           global_t_step,
                                                                                                           gripper1_target_x,
                                                                                                           gripper1_target_y,
                                                                                                           goal_rng)
        examples[current_record_traj_idx] = example
        percentages_positive.append(percentage_violation)

        # Save the data
        if current_record_traj_idx == args.trajs_per_file - 1:
            # Construct the dataset where each trajectory has been serialized into one big string
            # since tfrecords don't really support hierarchical data structures
            serialized_dataset = tensorflow.data.Dataset.from_tensor_slices((examples))

            end_traj_idx = i + args.start_idx_offset
            start_traj_idx = end_traj_idx - args.trajs_per_file + 1
            full_filename = os.path.join(full_output_directory, "traj_{}_to_{}.tfrecords".format(start_traj_idx, end_traj_idx))
            writer = tensorflow.data.experimental.TFRecordWriter(full_filename, compression_type=args.compression_type)
            writer.write(serialized_dataset)
            print("saved {}".format(full_filename))

            if args.verbose:
                mean_percentage_positive = np.mean(percentages_positive)
                print("Class balance: mean % positive: {}".format(mean_percentage_positive))

        if not args.verbose:
            print(".", end='')
            sys.stdout.flush()


def generate(args):
    rospy.init_node('collect_dynamics_data')

    n_state = ros_pycommon.get_n_state()
    rope_length = ros_pycommon.get_rope_length()

    assert args.trajs % args.trajs_per_file == 0, "num trajs must be multiple of {}".format(args.trajs_per_file)

    full_output_directory = random_environment_data_utils.data_directory(args.outdir, args.trajs)
    if not os.path.isdir(full_output_directory) and args.verbose:
        print(Fore.YELLOW + "Creating output directory: {}".format(full_output_directory) + Fore.RESET)
        os.mkdir(full_output_directory)

    local_env_params = LocalEnvParams(h_rows=args.local_env_rows, w_cols=args.local_env_cols, res=args.res)
    full_env_cols = int(args.env_w / args.res)
    full_env_rows = int(args.env_h / args.res)
    full_env_params = FullEnvParams(h_rows=full_env_rows, w_cols=full_env_cols, res=args.res)
    sim_params = SimParams(real_time_rate=args.real_time_rate,
                           max_step_size=args.max_step_size,
                           goal_padding=0.5,
                           move_obstacles=(not args.no_obstacles),
                           nudge=False)
    with open(pathlib.Path(full_output_directory) / 'hparams.json', 'w') as of:
        options = {
            'dt': args.dt,
            'max_step_size': args.max_step_size,
            'rope_length': rope_length,
            'local_env_params': local_env_params.to_json(),
            'full_env_params': full_env_params.to_json(),
            'sim_params': sim_params.to_json(),
            'compression_type': args.compression_type,
            'sequence_length': args.steps_per_traj,
            'n_state': n_state,
            'n_action': 2,
            'filter_free_space_only': False,
        }
        json.dump(options, of, indent=1)

    if args.seed is None:
        args.seed = np.random.randint(0, 10000)
    print(Fore.CYAN + "Using seed: {}".format(args.seed) + Fore.RESET)
    np.random.seed(args.seed)
    gazebo_rng = np.random.RandomState(args.seed)
    goal_rng = np.random.RandomState(args.seed)

    services = gazebo_utils.setup_gazebo_env(args.verbose, args.real_time_rate, args.max_step_size, True, None)

    generate_trajs(args, full_output_directory, services, gazebo_rng, goal_rng)


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=220, threshold=5000)
    tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.DEBUG)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("trajs", type=int, help='how many trajectories to collect')
    parser.add_argument("outdir")
    parser.add_argument('--dt', type=float, default=1.00, help='dt')
    parser.add_argument('--res', '-r', type=float, default=0.03, help='size of cells in meters')
    parser.add_argument('--env-w', type=float, default=6.0, help='full env w')
    parser.add_argument('--env-h', type=float, default=6.0, help='full env h')
    parser.add_argument('--local_env-cols', type=int, default=50, help='local env')
    parser.add_argument('--local_env-rows', type=int, default=50, help='local env')
    parser.add_argument("--steps-per-traj", type=int, default=100, help='steps per traj')
    parser.add_argument("--steps-per-target", type=int, default=25, help='steps before changing target')
    parser.add_argument("--start-idx-offset", type=int, default=0, help='offset TFRecord file names')
    parser.add_argument("--move-objects-every-n", type=int, default=16, help='rearrange objects every n trajectories')
    parser.add_argument("--no-obstacles", action='store_true', help='do not move obstacles')
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB', help='compression type')
    parser.add_argument("--trajs-per-file", type=int, default=256, help='trajs per file')
    parser.add_argument("--seed", '-s', type=int, help='seed')
    parser.add_argument("--real-time-rate", type=float, default=10, help='number of times real time')
    parser.add_argument("--max-step-size", type=float, default=0.01, help='seconds per physics step')
    parser.add_argument("--verbose", '-v', action="store_true", help='verbose')

    args = parser.parse_args()

    generate(args)


if __name__ == '__main__':
    main()
