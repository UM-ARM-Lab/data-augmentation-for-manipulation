from typing import Optional

import tensorflow as tf


def limit_gpu_mem(gigs: Optional[float]):
    gpus = tf.config.list_physical_devices('GPU')
    gpu = gpus[0]
    tf.config.experimental.set_memory_growth(gpu, True)
    if gigs is not None:
        config = [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * gigs)]
        tf.config.set_logical_device_configuration(gpu, config)
