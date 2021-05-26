import timeit
import tensorflow as tf


def main():
    n = 10000

    def _a():
        x = tf.transpose(tf.matmul(tf.random.normal([4, 4]), tf.random.normal([4, n])))[:, :3]

    def _b():
        x = tf.matmul(tf.random.normal([1, 4, 4]), tf.random.normal([n, 4, 1]))[:, :3, 0]

    print(timeit.timeit(_a, number=1000))
    print(timeit.timeit(_b, number=1000))


if __name__ == '__main__':
    main()
