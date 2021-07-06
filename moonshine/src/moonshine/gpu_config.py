import tensorflow as tf


def limit_gpu_mem(gigs: float):
    gpus = tf.config.list_physical_devices('GPU')
    gpu = gpus[0]
    config = [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * gigs)]
    tf.config.set_logical_device_configuration(gpu, config)
