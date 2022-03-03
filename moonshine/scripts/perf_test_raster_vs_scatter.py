import matplotlib.pyplot as plt
import tensorflow as tf

import rospy
from moonshine.raster_3d_tf import points_to_voxel_grid, points_to_voxel_grid_wrapped
from moonshine.simple_profiler import SimpleProfiler


def main():
    rospy.init_node("perf_test_raster_vs_scatter")

    # perf testing
    p = SimpleProfiler()

    rng = tf.random.Generator.from_seed(0)
    batch_size = 24
    s = 44
    n_points_in_component = 25
    m = 500
    batch_indices = tf.repeat(tf.range(batch_size, dtype=tf.int64), n_points_in_component, axis=0)
    state = rng.uniform([batch_size, n_points_in_component * 3], dtype=tf.float32)
    points = tf.reshape(state, [-1, 3])
    res = tf.constant([0.01] * batch_size)
    origin = tf.zeros([batch_size, 3], tf.float32)
    flat_res = tf.repeat(res, n_points_in_component, axis=0)
    flat_origin = tf.repeat(origin, n_points_in_component, axis=0)

    # indices = create_env_indices(s, s, s, batch_size)
    # print("raster 3d")
    # print(p.profile(m, raster_3d, state, indices.pixels, res, origin, s, s, s, 1000, batch_size))
    # print("raster 3d wrapped")
    # print(p.profile(m, raster_3d_wrapped, state, indices.pixels, res, origin, s, s, s, 1000, batch_size))

    print("points to vg")
    print(p.profile(m, points_to_voxel_grid, batch_indices, points, flat_res, flat_origin, s, s, s, batch_size))
    print("points to vg wrapped")
    print(p.profile(m, points_to_voxel_grid_wrapped, batch_indices, points, flat_res, flat_origin, s, s, s, batch_size))

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
