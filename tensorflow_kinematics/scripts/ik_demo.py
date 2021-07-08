import logging
import pathlib
from math import pi
from typing import Optional

import tensorflow as tf
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d as tfr
from tqdm import trange

import rospy
import urdf_parser_py.xml_reflection.core
from arc_utilities.ros_helpers import get_connected_publisher
from arc_utilities.tf2wrapper import TF2Wrapper
from link_bot_classifiers.robot_points import RobotVoxelgridInfo, batch_transform_robot_points
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from merrrt_visualization.rviz_animation_controller import RvizSimpleStepper
from moonshine.geometry import pairwise_squared_distances
from moonshine.moonshine_utils import reduce_mean_no_nan
from moonshine.simple_profiler import SimpleProfiler
from moonshine.tf_profiler_helper import TFProfilerHelper
from moveit_msgs.msg import DisplayRobotState
from sensor_msgs.msg import JointState
from tensorflow_kinematics.joint import SUPPORTED_ACTUATED_JOINT_TYPES
from tensorflow_kinematics.tree import Tree
from tensorflow_kinematics.urdf_utils import urdf_from_file, urdf_to_tree
from tf.transformations import quaternion_from_euler

IGNORE_COLLISIONS = [
    {"drive1", "drive2"},
    {"drive1", "torso"},
    {"drive2", "rightshoulder"},
    {"drive3", "rightshoulder"},
    {"drive3", "righttube"},
    {"drive4", "rightforearm"},
    {"drive4", "righttube"},
    {"drive41", "drive42"},
    {"drive41", "torso"},
    {"drive42", "leftshoulder"},
    {"drive43", "leftshoulder"},
    {"drive43", "lefttube"},
    {"drive44", "leftforearm"},
    {"drive44", "lefttube"},
    {"drive45", "drive46"},
    {"drive45", "leftforearm"},
    {"drive46", "leftwrist"},
    {"drive47", "end_effector_left"},
    {"drive47", "leftwrist"},
    {"drive5", "drive6"},
    {"drive5", "rightforearm"},
    {"drive56", "drive57"},
    {"drive56", "pedestal_link"},
    {"drive57", "torso"},
    {"drive6", "rightwrist"},
    {"drive7", "end_effector_right"},
    {"drive7", "rightwrist"},
    {"end_effector_left", "leftgripper_link"},
    {"end_effector_left", "leftgripper2_link"},
    {"end_effector_right", "rightgripper_link"},
    {"end_effector_right", "rightgripper2_link"},
    {"drive56", "torso"},
    {"drive57", "pedestal_link"},
    {"drive42", "torso"},
    {"drive2", "torso"},
    {"rightgripper2_link", "rightgripper_link"},
    {"leftgripper2_link", "leftgripper_link"},
    {"drive41", "drive43"},
    {"drive1", "drive3"},
    {"drive42", "drive43"},
    {"drive2", "drive3"},
    {"drive6", "drive7"},
    {"drive46", "drive47"},
    {"pedestal_link", "husky"},
    {"drive56", "husky"},
    {"drive57", "husky"},
    {"rightforearm", "righttube"},
    {"leftforearm", "lefttube"},
    {"realsense", "torso"},
    {"realsense", "drive41"},
    {"realsense", "drive1"},
]


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


def add_homo_batch(points):
    """

    Args:
        points: [n, 3]

    Returns: [1, n, 4, 1]

    """
    ones = tf.ones([points.shape[0], 1], dtype=tf.float32)
    points_homo = tf.concat([points, ones], axis=-1)
    points_homo_batch = points_homo[None, :, :, None]
    return points_homo_batch


def min_dists_and_indices(m):
    """

    Args:
        m:  [b,r,c]

    Returns: [b], [b], [b]

    """
    min_c_indices = tf.argmin(m, axis=-1)  # [b,r]
    min_along_c = tf.reduce_min(m, axis=-1)  # [b,r]
    min_r_indices = tf.argmin(min_along_c, axis=-1)  # [b]
    min_along_cr = tf.reduce_min(min_along_c, axis=-1)  # [b]
    min_c_indices = tf.gather(min_c_indices, min_r_indices, axis=-1, batch_dims=1)
    return min_along_cr, min_r_indices, min_c_indices


class HdtIK:

    def __init__(self,
                 urdf_filename: pathlib.Path,
                 scenario: ScenarioWithVisualization,
                 max_iters: int = 1000,
                 num_restarts: int = 100):
        self.avoid_env_collision = False
        self.avoid_self_collision = False
        self.urdf = urdf_from_file(urdf_filename.as_posix())
        self.scenario = scenario

        self.tree = urdf_to_tree(self.urdf)
        self.left_ee_name = 'left_tool'
        self.right_ee_name = 'right_tool'

        self.actuated_joint_names = list([j.name for j in self.urdf.joints if j.type in SUPPORTED_ACTUATED_JOINT_TYPES])
        self.n_actuated_joints = len(self.actuated_joint_names)

        self.robot_info = RobotVoxelgridInfo(joint_positions_key='!!!')

        self.max_iters = max_iters
        self.num_restarts = num_restarts
        self.initial_lr = 0.1
        self.orientation_weight = 0.0001
        self.jl_alpha = 0.1
        self.self_collision_alpha = 10.0
        self.collision_alpha = 50.0
        self.loss_threshold = 1e-4
        self.position_threshold = 1e-3
        self.orientation_threshold = 1e-3
        self.barrier_upper_lim = tf.square(0.04)  # stops repelling points from pushing after this distance
        self.barrier_scale = 0.05  # scales the gradients for the repelling points
        self.barrier_epsilon = 0.01
        self.log_cutoff = tf.math.log(self.barrier_scale * self.barrier_upper_lim + self.barrier_epsilon)

        self.robot_info.precompute_allowed_collidable_pairs(IGNORE_COLLISIONS)

        lr = tf.keras.optimizers.schedules.ExponentialDecay(self.initial_lr, int(self.max_iters / 10), 0.9)
        # lr = self.initial_lr
        # opt = tf.keras.optimizers.SGD(lr)
        self.optimizer = tf.keras.optimizers.Adam(lr, amsgrad=True)

        self.display_robot_state_pub = get_connected_publisher("display_robot_state", DisplayRobotState, queue_size=10)
        self.joint_states_viz_pub = rospy.Publisher("joint_states_viz", JointState, queue_size=10)
        self.tf2 = TF2Wrapper()

        self.p = SimpleProfiler()

    def solve(self,
              env_points,
              left_target_pose,
              right_target_pose,
              initial_value=None,
              viz=False,
              profiler_helper: Optional[TFProfilerHelper] = None):
        batch_size = left_target_pose.shape[0]
        full_batch_size = batch_size * self.num_restarts
        left_target_pose_repeated = tf.repeat(left_target_pose, self.num_restarts, axis=0)
        right_target_pose_repeated = tf.repeat(right_target_pose, self.num_restarts, axis=0)

        if initial_value is None:
            initial_value = self.sample_joint_positions(full_batch_size)
        q = tf.Variable(initial_value)

        converged = None
        loss_batch = None
        for iter in trange(self.max_iters):

            p = profiler_helper.start(batch_idx=iter, epoch=1)
            with tf.profiler.experimental.Trace('TraceContext', iter=iter):
                foo, _ = self.opt(q, env_points, left_target_pose_repeated, right_target_pose_repeated, full_batch_size,
                                  viz=viz)
            p.stop()
            left_pos_error, left_rot_error, right_pos_error, right_rot_error, jl_loss, loss_batch = foo
            conds = tf.stack([
                self.position_satisfied(left_pos_error),
                self.position_satisfied(right_pos_error),
                self.jl_satsified(jl_loss),
            ], axis=-1)
            converged_repeated = tf.reduce_all(conds, axis=-1)  # [b * num_restarts]
            converged = tf.reshape(converged_repeated, [batch_size, self.num_restarts])
            converged_any = tf.reduce_any(converged, axis=1)
            if tf.reduce_all(converged_any):
                break

        q = tf.reshape(q, [batch_size, self.num_restarts, -1])
        loss_batch = tf.reshape(loss_batch, [batch_size, self.num_restarts])
        score = tf.cast(converged, tf.float32) * -999 + loss_batch  # select the best solution out of the num_repeated
        best_solution_indices = tf.argmin(score, axis=1)
        best_q = tf.gather(q, best_solution_indices, axis=1, batch_dims=1)
        return best_q, converged_any

    def print_stats(self):
        print(self.p)

    def opt(self, q: tf.Variable, env_points, left_target_pose, right_target_pose, batch_size, viz: bool):
        with tf.GradientTape() as tape:
            self.p.start()
            foo, loss, viz_info = self.step(q, env_points, left_target_pose, right_target_pose, batch_size, viz)
            self.p.stop()
        gradient = tape.gradient([loss], [q])[0]

        if viz:
            self.viz_func(env_points, left_target_pose, right_target_pose, q, viz_info)

        self.optimizer.apply_gradients(grads_and_vars=[(gradient, q)])
        # delta = self.initial_lr * gradient
        # q.assign_sub(delta)

        return foo, gradient

    @tf.function
    def step(self, q: tf.Variable, env_points, left_target_pose, right_target_pose, batch_size, viz: bool):
        poses = self.tree.fk_no_recursion(q)
        jl_loss = self.compute_jl_loss(self.tree, q)
        left_ee_pose = poses[self.left_ee_name]
        right_ee_pose = poses[self.right_ee_name]
        left_pos_error, left_rot_error, left_pose_loss = self.compute_pose_loss(left_ee_pose, left_target_pose)
        right_pos_error, right_rot_error, right_pose_loss = self.compute_pose_loss(right_ee_pose, right_target_pose)

        losses = [
            left_pose_loss,
            right_pose_loss,
            jl_loss,
        ]

        if self.avoid_self_collision:
            self_collision_loss, self_collision_viz_info = self.compute_self_collision_loss(poses, batch_size, viz)
            losses.append(self_collision_loss)
        else:
            self_collision_viz_info = [None, None]

        if self.avoid_env_collision:
            collision_loss, collision_viz_info = self.compute_collision_loss(poses, env_points, batch_size, viz)
            losses.append(collision_loss)
        else:
            collision_viz_info = [None, None]

        loss_batch = tf.math.add_n(losses)
        loss = tf.reduce_mean(loss_batch)

        viz_info = [
            poses,
            self_collision_viz_info,
            collision_viz_info
        ]

        foo = [
            left_pos_error, left_rot_error,
            right_pos_error, right_rot_error,
            jl_loss,
            loss_batch,
        ]

        return foo, loss, viz_info

    def compute_self_collision_loss(self, poses, batch_size: int, viz: bool):
        nearest_points = []
        min_dists = []
        for (name1, l1), (name2, l2) in self.robot_info.allowed_collidable_pairs():
            # l1,l2 are in link frame, we need them in robot frame
            l1_robot_frame = self.link_points_robot_frame(poses, name1, l1)
            l2_robot_frame = self.link_points_robot_frame(poses, name2, l2)

            dists = pairwise_squared_distances(l1_robot_frame, l2_robot_frame)
            min_dist, l1_indices, l2_indices = min_dists_and_indices(dists)

            min_dists.append(min_dist)

            if viz:
                l1_nearest_point = tf.gather(l1_robot_frame, l1_indices, axis=1, batch_dims=1)
                l2_nearest_point = tf.gather(l2_robot_frame, l2_indices, axis=1, batch_dims=1)
                nearest_points.append((l1_nearest_point, l2_nearest_point))

        min_dists = tf.stack(min_dists, axis=1)
        if tf.size(nearest_points) > 0:
            nearest_points = tf.stack(nearest_points, axis=0)
            nearest_points = tf.transpose(nearest_points, [2, 0, 1, 3])
        viz_info = [nearest_points, min_dists]

        loss = self.self_collision_alpha * reduce_mean_no_nan(self.barrier_func(min_dists), axis=-1)
        return loss, viz_info

    def link_points_robot_frame(self, poses, name: str, link_points_link_frame):
        homo_batch = add_homo_batch(link_points_link_frame)
        transform = xm_to_44(poses[name])[:, None]
        robot_frame = tf.matmul(transform, homo_batch)[:, :, :3, 0]
        return robot_frame

    def compute_collision_loss(self, poses, env_points, batch_size, viz):
        link_transforms = self.reformat_link_transforms(poses)
        robot_points = batch_transform_robot_points(link_transforms, self.robot_info, batch_size)

        # compute the distance matrix between robot points and the environment points
        dists = pairwise_squared_distances(env_points, robot_points)
        min_dists = tf.reduce_min(dists, axis=-1)
        if viz:
            min_dists_indices = tf.argmin(dists, axis=-1)
            nearest_points = tf.gather(robot_points, min_dists_indices, axis=1, batch_dims=1)
        else:
            nearest_points = None

        viz_info = [nearest_points, min_dists]

        return self.collision_alpha * reduce_mean_no_nan(self.barrier_func(min_dists), axis=-1), viz_info

    def reformat_link_transforms(self, poses):
        link_to_robot_transforms = []
        for link_name in self.robot_info.link_names:
            link_to_robot_transform = xm_to_44(poses[link_name])
            link_to_robot_transforms.append(link_to_robot_transform)
        # [b, n_links, 4, 4, 1], links/order based on robot_info
        link_to_robot_transforms = tf.stack(link_to_robot_transforms, axis=1)
        return link_to_robot_transforms

    def barrier_func(self, min_dists_b):
        z = tf.math.log(self.barrier_scale * min_dists_b + self.barrier_epsilon)
        # of course this additive term doesn't affect the gradient, but it makes hyper-parameters more interpretable
        return tf.maximum(-z, -self.log_cutoff) + self.log_cutoff

    def compute_jl_loss(self, tree: Tree, q):
        return self.jl_alpha * compute_jl_loss(tree, q)

    def compute_pose_loss(self, xs, target_pose):
        pos_error, rot_error = compute_pose_loss(xs, target_pose)
        pose_loss = (1 - self.orientation_weight) * pos_error + self.orientation_weight * rot_error
        return pos_error, rot_error, pose_loss

    def viz_func(self, env_points, left_target_pose, right_target_pose, q, viz_info):
        poses, (self_nearest_points, self_min_dists), (nearest_points, min_dists) = viz_info
        b = 0

        if self.avoid_env_collision:
            p_b = []
            starts = []
            ends = []
            for env_point_i, nearest_point_i, min_d in zip(env_points[b], nearest_points[b], min_dists[b]):
                if min_d.numpy() < tf.square(self.barrier_upper_lim):
                    p_b.append(env_point_i[b].numpy())
                    p_b.append(nearest_point_i.numpy())
                    starts.append(env_point_i.numpy())
                    ends.append(nearest_point_i.numpy())
            self.scenario.plot_lines_rviz(starts, ends, label='nearest', color='red')

        if self.avoid_self_collision:
            p_b = []
            starts = []
            ends = []
            for self_nearest_points_i, min_d in zip(self_nearest_points[b], self_min_dists[b]):
                if min_d.numpy() < tf.square(self.barrier_upper_lim):
                    p_b.append(self_nearest_points_i[0].numpy())
                    p_b.append(self_nearest_points_i[1].numpy())
                    starts.append(self_nearest_points_i[0].numpy())
                    ends.append(self_nearest_points_i[1].numpy())
            self.scenario.plot_lines_rviz(starts, ends, label='self_nearest', color='white')

        self.plot_robot_and_targets(q, left_target_pose, right_target_pose, b)

        points = [pose.numpy()[b, :3] for pose in poses.values()]
        self.scenario.plot_points_rviz(points, label='fk', color='b', scale=0.005)

        self.scenario.plot_points_rviz(env_points[b].numpy(), label='env', color='magenta')

    def plot_robot_and_targets(self, q, left_target_pose, right_target_pose, b: int):
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

    def get_joint_names(self):
        return self.actuated_joint_names

    def get_num_joints(self):
        return self.n_actuated_joints

    @staticmethod
    def jl_satsified(jl_loss):
        return jl_loss < 1e-9

    def position_satisfied(self, pos_error):
        # pos_error is squared, so to make the threshold in meters we square it here
        return pos_error < tf.square(self.position_threshold)

    def sample_joint_positions(self, n: int):
        gen = tf.random.Generator.from_seed(0)
        joint_limits = self.tree.get_joint_limits()
        jl_low = joint_limits[:, 0][tf.newaxis]
        jl_high = joint_limits[:, 1][tf.newaxis]
        return gen.uniform([n, self.get_num_joints()], jl_low, jl_high, dtype=tf.float32)


def main():
    tf.get_logger().setLevel(logging.ERROR)
    rospy.init_node("ik_demo")

    def _on_error(_):
        pass

    urdf_parser_py.xml_reflection.core.on_error = _on_error

    urdf_filename = pathlib.Path("/home/peter/catkin_ws/src/hdt_robot/hdt_michigan_description/urdf/hdt_michigan.urdf")
    scenario = get_scenario("dual_arm_rope_sim_val_with_robot_feasibility_checking")
    ik_solver = HdtIK(urdf_filename, scenario)

    batch_size = 32
    viz = False
    profile = False

    gen = tf.random.Generator.from_seed(0)
    target_noise = gen.uniform([2, batch_size, 7],
                               [-1, -1, -1, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0],
                               dtype=tf.float32) * 0.1
    left_target_pose = tf.tile(target(-0.2, 0.55, 0.2, -pi / 2, 0, 0), [batch_size, 1]) + target_noise[0]
    right_target_pose = tf.tile(target(0.2, 0.55, 0.22, -pi / 2 + 0.5, -pi, 0), [batch_size, 1]) + target_noise[1]
    # right_target_pose = tf.tile(target(0.0, 0.0, 0.0, 0, -pi, 0), [batch_size, 1])
    o = tf.constant([[[-0.25, 0.2, 0.2]]], tf.float32)
    env_points = tf.random.uniform([batch_size, 100, 3], -0.1, 0.1, dtype=tf.float32) + o

    # gen = tf.random.Generator.from_seed(0)
    # initial_noise = gen.uniform([batch_size, ik_solver.get_num_joints()], -1, 1, dtype=tf.float32) * 0.1
    # initial_value = tf.zeros([batch_size, ik_solver.get_num_joints()], dtype=tf.float32) + initial_noise

    logdir = "ik_demo_logdir"
    if profile:
        profile_arg = (1, 29)
    else:
        profile_arg = None
    h = TFProfilerHelper(profile_arg=profile_arg, train_logdir=logdir)
    q, converged = ik_solver.solve(env_points=env_points,
                                   left_target_pose=left_target_pose,
                                   right_target_pose=right_target_pose,
                                   viz=viz,
                                   profiler_helper=h)

    ik_solver.get_joint_names()
    print(f'{converged=}')
    ik_solver.print_stats()

    stepper = RvizSimpleStepper()
    for b in range(batch_size):
        ik_solver.plot_robot_and_targets(q, left_target_pose, right_target_pose, b=b)
        stepper.step()


if __name__ == '__main__':
    main()
