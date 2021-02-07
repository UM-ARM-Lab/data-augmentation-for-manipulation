from unittest import TestCase

import rospy
from link_bot_gazebo_python.gazebo_services import GazeboServices
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.grid_utils import point_to_idx_3d_in_env
from link_bot_pycommon.pycommon import longest_reconverging_subsequence, trim_reconverging
from link_bot_pycommon.ros_pycommon import make_movable_object_services
from link_bot_pycommon.get_occupancy import get_environment_for_extents_3d


class Test(TestCase):
    def test_start_and_end_of_max_consecutive_zeros(self):
        # contains no reconverging
        self.assertEqual(longest_reconverging_subsequence([]), (0, 0))
        self.assertEqual(longest_reconverging_subsequence([1]), (0, 0))
        self.assertEqual(longest_reconverging_subsequence([0]), (0, 0))
        self.assertEqual(longest_reconverging_subsequence([0, 0]), (0, 0))
        self.assertEqual(longest_reconverging_subsequence([1, 0]), (0, 0))
        self.assertEqual(longest_reconverging_subsequence([1, 0, 0]), (0, 0))
        # contains reconverging
        self.assertEqual(longest_reconverging_subsequence([0, 1, 0]), (0, 1))
        self.assertEqual(longest_reconverging_subsequence([0, 1, 0, 0]), (0, 1))
        self.assertEqual(longest_reconverging_subsequence([0, 0, 1]), (0, 2))
        self.assertEqual(longest_reconverging_subsequence([0, 0, 1, 0]), (0, 2))
        self.assertEqual(longest_reconverging_subsequence([1, 0, 0, 1]), (1, 3))
        self.assertEqual(longest_reconverging_subsequence([0, 0, 1, 0, 0, 0, 1, 0]), (3, 6))

    def test_trim_reconverging(self):
        self.assertEqual(trim_reconverging([1, 0, 1]), (0, 3))
        self.assertEqual(trim_reconverging([1, 0, 0, 1]), (0, 4))
        self.assertEqual(trim_reconverging([1, 0, 0, 1, 1]), (0, 5))
        self.assertEqual(trim_reconverging([1, 0, 0, 1]), (0, 4))
        self.assertEqual(trim_reconverging([1, 0, 0, 1, 1, 0]), (0, 5))
        self.assertEqual(trim_reconverging([1, 0, 1, 0, 0, 1, 1]), (2, 7))
        self.assertEqual(trim_reconverging([1, 0, 0, 0, 1]), (0, 5))
        self.assertEqual(trim_reconverging([1, 0, 1, 0, 1]), (0, 3))  # tie break goes to the first occurrence
        self.assertEqual(trim_reconverging([1, 0, 1, 0, 0, 1]), (2, 6))
        self.assertEqual(trim_reconverging([1, 1, 0, 0, 1, 1]), (0, 6))
        self.assertEqual(trim_reconverging([1, 0, 0, 1, 1, 0, 0, 0, 1, 0]), (3, 9))
        #                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9,


class TestOccupancy(TestCase):
    def test_occupancy(self):
        rospy.init_node('test_occupancy')
        service_provider = GazeboServices()
        movable_objects_services = {
            "moving_box1": make_movable_object_services("moving_box1")
        }
        xs = [-1, 1, 1, -1, -1]
        ys = [-1, -1, 1, 1, -1]
        res = 0.1
        extent = [-1, 1, -1, 1, 0, 1]
        for x_i, y_i in zip(xs, ys):
            object_positions = {
                "moving_box1": [x_i, y_i]
            }
            ExperimentScenario.move_objects_to_positions(movable_objects_services, object_positions, timeout=150)
            environment = get_environment_for_extents_3d(extent=extent, res=res, service_provider=service_provider,
                                                         robot_name="test")
            # scale down to avoid out of bounds on the edges
            row_i, col_i, channel_i = point_to_idx_3d_in_env(x=0.99 * x_i, y=0.99 * y_i, z=0.01, environment=environment)
            occupied = environment['env'][row_i, col_i, channel_i] > 0
            self.assertTrue(occupied)

            row_i, col_i, channel_i = point_to_idx_3d_in_env(x=0 * x_i, y=0 * y_i, z=0.01, environment=environment)
            occupied = environment['env'][row_i, col_i, channel_i] > 0
            self.assertFalse(occupied)
