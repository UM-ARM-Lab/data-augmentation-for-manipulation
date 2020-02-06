#!/usr/bin/env python
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import rospy
import tensorflow as tf

from link_bot_gazebo import gazebo_utils
from link_bot_gazebo.gazebo_utils import get_local_occupancy_data, GazeboServices, get_occupancy_data
from link_bot_gazebo.srv import LinkBotStateRequest
from link_bot_planning import model_utils, classifier_utils
from link_bot_classifiers.visualization import plot_classifier_data
from link_bot_pycommon.args import my_formatter

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(gpu_options=gpu_options)
tf.enable_eager_execution(config=config)


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=250)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("fwd_model_dir", help="load this saved forward model file", type=pathlib.Path)
    parser.add_argument("fwd_model_type", choices=['nn', 'gp', 'llnn', 'rigid', 'obs'], default='nn')
    parser.add_argument("classifier_model_dir", help="classifier", type=pathlib.Path)
    parser.add_argument("classifier_model_type", choices=['collision', 'none', 'raster'], default='raster')
    parser.add_argument('--res', '-r', type=float, default=0.03, help='size of cells in meters')
    parser.add_argument('--no-plot', action='store_true', help="don't show plots, useful for debugging")

    args = parser.parse_args()

    # use forward model to predict given the input action
    fwd_model, _ = model_utils.load_generic_model(args.fwd_model_dir, args.fwd_model_type)
    classifier_model = classifier_utils.load_generic_model(args.classifier_model_dir, args.classifier_model_type)
    local_env_params = fwd_model.local_env_params
    full_env_params = fwd_model.full_env_params
    cols = local_env_params.w_cols
    rows = local_env_params.h_rows

    rospy.init_node('test_classifier_from_gazebo')

    services = GazeboServices()

    state_req = LinkBotStateRequest()
    link_bot_state = services.get_state(state_req)
    head_idx = link_bot_state.link_names.index("head")
    head_point = link_bot_state.points[head_idx]
    head_point = np.array([head_point.x, head_point.y])
    local_env_data = get_local_occupancy_data(rows, cols, args.res, center_point=head_point, services=services)

    full_env_data = get_occupancy_data(env_w=full_env_params.w,
                                       env_h=full_env_params.h,
                                       res=full_env_params.res,
                                       services=services)

    state = np.expand_dims(gazebo_utils.points_to_config(link_bot_state.points), axis=0)

    test_inputs = [
        # (.25, 0),
        # (.25, 45),
        # (.25, 90),
        # (.25, 135),
        # (.25, 180),
        # (.25, 225),
        # (.25, 270),
        # (.25, 325),
        (.15, 135),
        (.15, 90),
        (.15, 45),
        (.15, 180),
        (0, 0),
        (.15, 0),
        (.15, 225),
        (.15, 270),
        (.15, 315),
        # (.05, 0),
        # (.05, 45),
        # (.05, 90),
        # (.05, 135),
        # (.05, 180),
        # (.05, 225),
        # (.05, 270),
        # (.05, 315),
    ]

    fig, axes = plt.subplots(3, 3)
    for k, (v, theta_deg) in enumerate(test_inputs):
        theta_rad = np.deg2rad(theta_deg)
        vx = np.cos(theta_rad) * v
        vy = np.sin(theta_rad) * v
        action = np.array([[[vx, vy]]])

        next_state = fwd_model.predict(full_envs=[full_env_data.data],
                                       full_env_origins=[full_env_data.origin],
                                       resolution_s=[full_env_data.resolution],
                                       state=state,
                                       actions=action)
        next_state = np.reshape(next_state, [2, 1, -1])[1]

        # classifier_model.show = True
        accept_probability = classifier_model.predict([local_env_data], state, next_state, action)[0]
        prediction = 1 if accept_probability > 0.5 else 0
        title = 'P(accept) = {:04.3f}%'.format(100 * accept_probability)

        print("v={:04.3f}m/s theta={:04.3f}deg    p(accept)={:04.3f}".format(v, theta_deg, accept_probability))
        if not args.no_plot:
            i, j = np.unravel_index(k, [3, 3])
            plot_classifier_data(ax=axes[i, j],
                                 planned_env=local_env_data.data,
                                 planned_env_extent=local_env_data.extent,
                                 planned_state=state[0],
                                 planned_next_state=next_state[0],
                                 planned_env_origin=local_env_data.origin,
                                 res=local_env_data.resolution,
                                 state=None,
                                 next_state=None,
                                 title=title,
                                 actual_env=None,
                                 actual_env_extent=None,
                                 label=prediction)

    if not args.no_plot:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
