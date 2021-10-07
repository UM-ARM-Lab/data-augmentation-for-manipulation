from functools import lru_cache
from io import BytesIO

import tensorflow as tf

def bytes_to_ros_msg(bytes, msg_type: type):
    msg = msg_type()
    msg.deserialize(bytes)
    return msg


def ros_msg_to_bytes_tensor(msg):
    buff = BytesIO()
    msg.serialize(buff)
    serialized_bytes = buff.getvalue()
    return tf.convert_to_tensor(serialized_bytes)


