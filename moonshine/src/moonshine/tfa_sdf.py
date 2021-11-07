import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_addons.image import euclidean_dist_transform
from tensorflow_felzenszwalb_edt import edt1d

import ros_numpy
import rospy
from arc_utilities import ros_init
from moonshine.matrix_operations import shift_and_pad
from moonshine.moonshine_utils import repeat_tensor
from moonshine.simple_profiler import SimpleProfiler
from sensor_msgs.msg import PointCloud2


def get_grid_points(origin_point, res, shape):
    indices = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.stack(indices, axis=-1)
    points = (indices * res) - origin_point
    return points


def visualize_sdf(pub, sdf: np.ndarray, shape, res, origin_point):
    points = get_grid_points(origin_point, res, shape)
    list_of_tuples = [(p[0], p[1], p[2], d) for p, d in zip(points.reshape([-1, 3]), sdf.flatten())]
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('distance', np.float32)]
    np_record_array = np.array(list_of_tuples, dtype=dtype)
    msg = ros_numpy.msgify(PointCloud2, np_record_array, frame_id='world', stamp=rospy.Time.now())
    pub.publish(msg)


def build_sdf_2d(vg, res, origin_point):
    binary_vg_batch = tf.cast([vg], tf.uint8)
    filled_distance_field = euclidean_dist_transform(binary_vg_batch)
    empty_distance_field = tf.maximum(euclidean_dist_transform(1 - binary_vg_batch) - 1, 0)
    distance_field = empty_distance_field + -filled_distance_field

    plt.figure()
    plt.imshow(distance_field[0, :, :, 0])
    plt.yticks(range(vg.shape[0]))
    plt.xticks(range(vg.shape[1]))
    plt.show()


def compute_sdf_and_gradient_batch(vg, res):
    sdf = build_sdf_3d(vg, res)
    sdf_grad = build_grad_3d(sdf)
    return sdf, sdf_grad


def build_grad_3d_partial(sdf, axis):
    right = shift_and_pad(sdf, shift=1, pad_value=-100, axis=axis)
    left = shift_and_pad(sdf, shift=-1, pad_value=-100, axis=axis)
    partial = tf.sign(left - right)
    return partial


def build_grad_3d(sdf):
    """

    Args:
        sdf: [b, h, w, c]

    Returns: [b, h, w, c, 3]

    """
    h_grad = build_grad_3d_partial(sdf, 1)
    w_grad = build_grad_3d_partial(sdf, 2)
    c_grad = build_grad_3d_partial(sdf, 3)

    # swapping w and h here because we want x,y,z and w=x, h=y, c=z
    grad = tf.stack([w_grad, h_grad, c_grad], axis=-1)

    return grad


def build_sdf_3d(vg, res):
    """

    Args:
        vg: [b, h, w ,c] of type float32
        res: [b]

    Returns: [b, h, w, c]

    """
    # NOTE: this is how the morphological EDT works, you first scale everything by a big number
    s = np.sum(np.array(vg.shape) ** 2)
    filled_vg = vg * s
    filled_distance_field = edt1d(filled_vg, 1)[0]
    filled_distance_field = edt1d(filled_distance_field, 2)[0]
    filled_distance_field = edt1d(filled_distance_field, 3)[0]
    filled_distance_field = tf.sqrt(filled_distance_field)

    empty_vg = (1 - vg) * s
    empty_distance_field = edt1d(empty_vg, 1)[0]
    empty_distance_field = edt1d(empty_distance_field, 2)[0]
    empty_distance_field = edt1d(empty_distance_field, 3)[0]
    empty_distance_field = tf.sqrt(empty_distance_field)

    distance_field = empty_distance_field + -filled_distance_field

    distance_field_meters = distance_field * res[..., None, None, None]
    return distance_field_meters


@ros_init.with_ros("tfa_sdf")
def main():
    sdf_pub = rospy.Publisher("sdf", PointCloud2, queue_size=10)

    batch_size = 32
    res = repeat_tensor(0.005, batch_size, 0, True)
    shape = [batch_size, 100, 100, 20]
    origin_point = np.zeros(3, np.float32)
    origin_point = repeat_tensor(origin_point, batch_size, 0, True)

    vg = np.zeros(shape, np.float32)
    vg[0, :5, :5, :5] = 1.0

    # sdf = build_sdf_3d(vg, res, origin_point)
    sdf, grad = compute_sdf_and_gradient_batch(vg, res)

    for b in range(1):
        visualize_sdf(sdf_pub, sdf[0].numpy(), shape, res[0], origin_point[0])


def perf():
    p = SimpleProfiler()
    batch_size = 32
    res = repeat_tensor(0.005, batch_size, 0, True)
    shape = [batch_size, 100, 100, 20]
    origin_point = np.zeros(3, np.float32)
    origin_point = repeat_tensor(origin_point, batch_size, 0, True)

    vg = np.zeros(shape, np.float32)
    vg[:, :5, :5, :5] = 1.0

    def _sdf():
        sdf = build_sdf_3d(vg, res)

    print(p.profile(100, _sdf))


if __name__ == '__main__':
    main()
    # perf()
