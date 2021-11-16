#!/usr/bin/env python
import argparse
import pathlib

import colorama
import numpy as np

import rospy
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.moonshine_utils import remove_batch
from moonshine.numpify import numpify
from state_space_dynamics.train_test_dynamics import viz_dataset
from visualization_msgs.msg import Marker


class VizLatentSpace:

    def __init__(self):
        self.first_latent_state = None
        self.idx = 0
        self.latent_state_pub = rospy.Publisher("latent_state_viz", Marker, queue_size=10)

    def viz_func(self, batch, outputs, test_dataset: DynamicsDatasetLoader):
        """ we assume batch size of 1 """
        test_dataset.scenario.plot_environment_rviz(remove_batch(batch))
        anim = RvizAnimationController(np.arange(test_dataset.steps_per_traj))
        while not anim.done:
            t = anim.t()

            if self.first_latent_state is None:
                self.first_latent_state = outputs['z'][0, 0]
                pass

            m = Marker()
            m.header.frame_id = 'world'
            m.header.stamp = rospy.Time.now()
            m.type = Marker.SPHERE
            m.action = Marker.MODIFY
            m.scale.x = 0.01
            m.scale.y = 0.01
            m.scale.z = 0.01
            m.color.r = 0.8
            m.color.g = 0.2
            m.color.b = 0.8
            m.color.a = 0.8
            m.id = self.idx
            m.ns = 'latent state'
            m.pose.position.x = (outputs['z'][0, 0, 0] - self.first_latent_state[0]) * 10
            m.pose.position.y = (outputs['z'][0, 0, 1] - self.first_latent_state[1]) * 10
            m.pose.position.z = (outputs['z'][0, 0, 2] - self.first_latent_state[2]) * 10
            m.pose.orientation.w = 1
            self.latent_state_pub.publish(m)

            e_t = numpify(remove_batch(test_dataset.scenario.index_time_batched_predicted(batch, t)))
            test_dataset.scenario.plot_state_rviz(e_t, label='actual', color='red')
            test_dataset.scenario.plot_action_rviz(e_t, e_t, color='gray')

            self.idx += 1

            anim.step()


def main():
    colorama.init(autoreset=True)

    np.set_printoptions(linewidth=250, precision=4, suppress=True, threshold=10000)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('checkpoint', type=pathlib.Path)
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'val'], default='test')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    rospy.init_node('viz_obs')
    v = VizLatentSpace()
    viz_dataset(dataset_dirs=args.dataset_dirs,
                checkpoint=args.checkpoint,
                mode=args.mode,
                viz_func=v.viz_func)


if __name__ == '__main__':
    main()
