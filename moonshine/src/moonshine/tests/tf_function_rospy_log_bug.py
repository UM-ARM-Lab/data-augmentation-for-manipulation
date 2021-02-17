import tensorflow as tf

import rospy


def my_nontf_function(x):
    return tf.square(x)


@tf.function
def my_tf_function(x):
    return tf.square(x)


def my_nontf_function_with_logging(x):
    rospy.logerr_once("my_nontf_function_with_logging")
    return tf.square(x)


@tf.function
def my_tf_function_with_logging(x):
    rospy.logerr_once("my_ntf_function_with_logging")
    return tf.square(x)


if __name__ == '__main__':
    x = tf.range(10)
    my_nontf_function(x)
    my_nontf_function_with_logging(x)
    my_tf_function(x)
    my_tf_function_with_logging(x)
    print("done.")
