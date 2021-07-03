import logging
import pathlib
from math import pi

import tensorflow as tf
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d as tfr
from tqdm import trange

import rospy
import urdf_parser_py.xml_reflection.core
from arc_utilities.ros_helpers import get_connected_publisher
from arc_utilities.tf2wrapper import TF2Wrapper
from geometry_msgs.msg import Point
from link_bot_classifiers.robot_points import RobotVoxelgridInfo
from moonshine.geometry import pairwise_squared_distances
from moonshine.moonshine_utils import repeat_tensor, reduce_mean_no_nan
from moonshine.simple_profiler import SimpleProfiler
from moveit_msgs.msg import DisplayRobotState
from sensor_msgs.msg import JointState
from tf.transformations import quaternion_from_euler
from tf_robot_learning.kinematic.joint import SUPPORTED_ACTUATED_JOINT_TYPES
from tf_robot_learning.kinematic.tree import Tree
from tf_robot_learning.kinematic.urdf_utils import urdf_from_file, urdf_to_tree
from visualization_msgs.msg import Marker


def orientation_error_quat(q1, q2):
    # NOTE: I don't know of a correct & smooth matrix -> quaternion implementation
    # https://www.ri.cmu.edu/pub_files/pub4/kuffner_james_2004_1/kuffner_james_2004_1.pdf
    # https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf
    return 1 - tf.square(tf.einsum('bi,bi->b', q1, q2))


def orientation_error_mat(r1, r2):
    # https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf
    I = tf.expand_dims(tf.eye(3), 0)
    R_Rt = tf.matmul(r1, tf.transpose(r2, [0, 2, 1]))
    return tf.linalg.norm(I - R_Rt, ord='fro', axis=[-2, -1])


def orientation_error_mat2(r1, r2):
    # NOTE: not differentiable
    # https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf
    R_Rt = tf.matmul(r1, tf.transpose(r2, [0, 2, 1]))
    log = tf.cast(tf.linalg.logm(tf.cast(R_Rt, tf.complex64)), tf.float32)
    return tf.linalg.norm(log, ord='fro', axis=[-2, -1])


def compute_pose_loss(ee_pose, target_pose):
    position = ee_pose[:, :3]
    orientation = tf.reshape(ee_pose[:, 3:], [-1, 3, 3])
    target_position = target_pose[:, :3]
    target_quat = target_pose[:, 3:]
    target_orientation = tfr.from_quaternion(target_quat)
    _orientation_error = orientation_error_mat(target_orientation, orientation)
    position_error = tf.reduce_sum(tf.square(position - target_position), axis=-1)

    return position_error, _orientation_error


def compute_jl_loss(tree: Tree, q):
    joint_limits = tree.get_joint_limits()
    jl_low = joint_limits[:, 0][tf.newaxis]
    jl_high = joint_limits[:, 1][tf.newaxis]
    low_error = tf.math.maximum(jl_low - q, 0)
    high_error = tf.math.maximum(q - jl_high, 0)
    jl_errors = tf.math.maximum(low_error, high_error)
    jl_loss = tf.reduce_sum(jl_errors, axis=-1)
    return jl_loss


def target(x, y, z, roll, pitch, yaw):
    return tf.cast(tf.expand_dims(tf.concat([[x, y, z], quaternion_from_euler(roll, pitch, yaw)], 0), 0), tf.float32)


def xm_to_44(xm):
    """
    Args:
        xm: [b, 12]
    Returns: [b, 4, 4]
    """
    b = xm.shape[0]
    p = xm[:, :3]  # [b,3]
    p = p[:, :, None]  # [b,3,1]
    r = xm[:, 3:]  # [b,9]
    r33 = tf.reshape(r, [-1, 3, 3])  # [b,3,3]
    m34 = tf.concat([r33, p], axis=-1)  # [b,3,4]
    h = tf.constant([[0, 0, 0, 1]], dtype=tf.float32)  # [1,4]
    h = tf.tile(h[None], [b, 1, 1])  # [b,1,4]
    m44 = tf.concat([m34, h], axis=-2)  # [b,4,4]
    return m44


class HdtIK:

    def __init__(self, urdf_filename: pathlib.Path, max_iters: int = 5000):
        self.urdf = urdf_from_file(urdf_filename.as_posix())

        self.tree = urdf_to_tree(self.urdf)
        self.left_ee_name = 'left_tool'
        self.right_ee_name = 'right_tool'

        self.actuated_joint_names = list([j.name for j in self.urdf.joints if j.type in SUPPORTED_ACTUATED_JOINT_TYPES])
        self.n_actuated_joints = len(self.actuated_joint_names)

        self.robot_info = RobotVoxelgridInfo(joint_positions_key='!!!')

        self.max_iters = max_iters
        self.initial_lr = 0.05
        self.theta = 0.992
        self.jl_alpha = 0.1
        self.loss_threshold = 1e-4
        self.barrier_upper_lim = tf.square(0.06)  # stops repelling points from pushing after this distance
        self.barrier_scale = 0.05  # scales the gradients for the repelling points
        self.barrier_epsilon = 0.01
        self.log_cutoff = tf.math.log(self.barrier_scale * self.barrier_upper_lim + self.barrier_epsilon)

        lr = tf.keras.optimizers.schedules.ExponentialDecay(self.initial_lr, int(self.max_iters / 10), 0.9)
        # lr = self.initial_lr
        # opt = tf.keras.optimizers.SGD(lr)
        self.optimizer = tf.keras.optimizers.Adam(lr, amsgrad=True)

        self.display_robot_state_pub = get_connected_publisher("display_robot_state", DisplayRobotState, queue_size=10)
        self.point_pub = get_connected_publisher("point", Marker, queue_size=10)
        self.joint_states_viz_pub = rospy.Publisher("joint_states_viz", JointState, queue_size=10)
        self.tf2 = TF2Wrapper()

        self.p = SimpleProfiler()

    def solve(self, env_points, left_target_pose, right_target_pose, initial_value=None, viz=False):
        if initial_value is None:
            batch_size = left_target_pose.shape[0]
            initial_value = tf.zeros([batch_size, self.get_num_joints()], dtype=tf.float32)
        q = tf.Variable(initial_value)

        converged = False
        for _ in trange(self.max_iters):
            loss, gradients, viz_info = self.opt(q, env_points, left_target_pose, right_target_pose)
            if loss < self.loss_threshold:
                converged = True
                break

            if viz:
                self.viz_func(left_target_pose, right_target_pose, q, viz_info)

        return q, converged

    def print_stats(self):
        print(self.p)

    def opt(self, q, env_points, left_target_pose, right_target_pose):
        with tf.GradientTape() as tape:
            self.p.start()
            loss, viz_info = self.step(q, env_points, left_target_pose, right_target_pose)
            self.p.stop()
        gradients = tape.gradient([loss], [q])
        self.optimizer.apply_gradients(grads_and_vars=zip(gradients, [q]))
        return loss, gradients, viz_info

    # @tf.function
    def step(self, q, env_points, left_target_pose, right_target_pose):
        poses = self.tree.fk(q)
        jl_loss = self.compute_jl_loss(self.tree, q)
        left_ee_pose = poses[self.left_ee_name]
        right_ee_pose = poses[self.right_ee_name]
        left_pose_loss = self.compute_pose_loss(left_ee_pose, left_target_pose)
        right_pose_loss = self.compute_pose_loss(right_ee_pose, right_target_pose)

        # collision_loss, collision_viz_info = self.compute_collision_loss(poses, env_points)

        losses = [
            left_pose_loss,
            right_pose_loss,
            jl_loss,
            # collision_loss,
        ]
        loss = tf.reduce_mean(tf.math.add_n(losses))

        viz_info = [poses]
        # viz_info.extend(collision_viz_info)

        return loss, viz_info

    def compute_collision_loss(self, poses, env_points):
        # compute robot points given q
        link_to_robot_transforms = []
        for link_name in self.robot_info.link_names:
            link_to_robot_transform = xm_to_44(poses[link_name])
            link_to_robot_transforms.append(link_to_robot_transform)
        # [b, n_links, 4, 4, 1], links/order based on robot_info
        link_to_robot_transforms = tf.stack(link_to_robot_transforms, axis=1)
        links_to_robot_transform_batch = tf.repeat(link_to_robot_transforms, self.robot_info.points_per_links,
                                                   axis=1)
        batch_size = env_points.shape[0]
        points_link_frame_homo_batch = repeat_tensor(self.robot_info.points_link_frame, batch_size, 0, True)
        points_robot_frame_homo_batch = tf.matmul(links_to_robot_transform_batch, points_link_frame_homo_batch)
        points_robot_frame_batch = points_robot_frame_homo_batch[:, :, :3, 0]  # [b, n_env_points, n_robot_points]

        # compute the distance matrix between robot points and the environment points
        dists = pairwise_squared_distances(env_points, points_robot_frame_batch)
        # FIXME: add visualization
        min_dists_indices = tf.argmin(dists, axis=-1)
        min_dist_robot_points = tf.gather(points_robot_frame_batch, min_dists_indices, axis=-1)
        min_dists = tf.reduce_min(dists, axis=-1)

        viz_info = [min_dists_indices]

        return reduce_mean_no_nan(self.barrier_func(min_dists), axis=-1), viz_info

    def barrier_func(self, min_dists_b):
        z = tf.math.log(self.barrier_scale * min_dists_b + self.barrier_epsilon)
        # of course this additive term doesn't affect the gradient, but it makes hyper-parameters more interpretable
        return tf.maximum(-z, -self.log_cutoff) + self.log_cutoff

    def compute_jl_loss(self, tree: Tree, q):
        return self.jl_alpha * compute_jl_loss(tree, q)

    def compute_pose_loss(self, xs, target_pose):
        pos_error, rot_error = compute_pose_loss(xs, target_pose)
        pose_loss = self.theta * pos_error + (1 - self.theta) * rot_error
        return pose_loss

    def viz_func(self, left_target_pose, right_target_pose, q, viz_info):
        poses, = viz_info
        b = 0
        self.tf2.send_transform(left_target_pose[b, :3].numpy().tolist(),
                                left_target_pose[b, 3:].numpy().tolist(),
                                parent='world', child='left_target')
        self.tf2.send_transform(right_target_pose[b, :3].numpy().tolist(),
                                right_target_pose[b, 3:].numpy().tolist(),
                                parent='world', child='right_target')

        robot_state_dict = {}
        for name, pose in zip(self.actuated_joint_names, q[b].numpy().tolist()):
            robot_state_dict[name] = pose

        robot = DisplayRobotState()
        robot.state.joint_state.name = robot_state_dict.keys()
        robot.state.joint_state.position = robot_state_dict.values()
        robot.state.joint_state.header.stamp = rospy.Time.now()
        self.display_robot_state_pub.publish(robot)
        self.joint_states_viz_pub.publish(robot.state.joint_state)

        msg = Marker()
        msg.header.frame_id = 'world'
        msg.header.stamp = rospy.Time.now()
        msg.id = 0
        msg.type = Marker.SPHERE_LIST
        msg.action = Marker.ADD
        msg.pose.orientation.w = 1
        scale = 0.01
        msg.scale.x = scale
        msg.scale.y = scale
        msg.scale.z = scale
        msg.color.r = 1
        msg.color.a = 1
        msg.points = []
        for pose in poses.values():
            position = pose.numpy()[b, :3]
            p = Point(x=position[0], y=position[1], z=position[2])
            msg.points.append(p)

        self.point_pub.publish(msg)

    def get_joint_names(self):
        return self.actuated_joint_names

    def get_num_joints(self):
        return self.n_actuated_joints


def main():
    tf.get_logger().setLevel(logging.ERROR)
    rospy.init_node("ik_demo")

    def _on_error(_):
        pass

    urdf_parser_py.xml_reflection.core.on_error = _on_error

    urdf_filename = pathlib.Path("/home/peter/catkin_ws/src/hdt_robot/hdt_michigan_description/urdf/hdt_michigan.urdf")
    ik_solver = HdtIK(urdf_filename, max_iters=500)

    batch_size = 32
    viz = True

    left_target_pose = tf.tile(target(-0.2, 0.3, 0.3, 0, -pi / 2, -pi / 2), [batch_size, 1])
    right_target_pose = tf.tile(target(0.4, 0.6, 0.4, -pi / 2, -pi / 2, 0), [batch_size, 1])
    env_points = tf.random.uniform([batch_size, 10, 3], -1, 1, dtype=tf.float32)

    initial_value = tf.zeros([batch_size, ik_solver.get_num_joints()], dtype=tf.float32)
    q, converged = ik_solver.solve(env_points=env_points,
                                   left_target_pose=left_target_pose,
                                   right_target_pose=right_target_pose,
                                   viz=viz,
                                   initial_value=initial_value)
    ik_solver.get_joint_names()
    print(f'{converged=}')
    ik_solver.print_stats()


if __name__ == '__main__':
    main()
