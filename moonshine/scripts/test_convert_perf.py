import timeit
import tensorflow as tf


def main():
    x = tf.random.normal([32, 10, 75])

    def _a():
        tf.convert_to_tensor(x)

    def _b():
        tf.constant(x)

    print(timeit.timeit(_a, number=10000))
    print(timeit.timeit(_b, number=10000))


if __name__ == '__main__':
    main()
