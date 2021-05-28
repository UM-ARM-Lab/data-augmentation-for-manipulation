import tensorflow as tf

import rospy
from moonshine.indexing import slice_along_axis
from moonshine.simple_profiler import SimpleProfiler


def main():
    rospy.init_node("perf_gather_vs_slices")

    p = SimpleProfiler()

    x = tf.random.uniform([100, 100, 100])
    start = 30
    end = 90

    def _a():
        return tf.gather(x, tf.range(start, end), axis=0)

    def _b():
        return tf.gather(x, tf.range(start, end), axis=1)

    def _c():
        return x[start:end]

    def _d():
        return x[:, start:end]

    def _e():
        return slice_along_axis(x, start, end, 0)

    def _f():
        return slice_along_axis(x, start, end, 1)

    print("gather")
    print(p.profile(10000, _a))
    print(p.profile(10000, _b))
    print("slice")
    print(p.profile(10000, _c))
    print(p.profile(10000, _d))
    print("my slice")
    print(p.profile(10000, _e))
    print(p.profile(10000, _f))


if __name__ == '__main__':
    main()
