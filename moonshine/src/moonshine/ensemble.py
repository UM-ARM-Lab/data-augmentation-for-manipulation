from typing import List, Callable

import numpy as np
import tensorflow as tf

from moonshine.tensorflow_utils import sequence_of_dicts_to_dict_of_tensors


class Ensemble2:
    def __init__(self, elements, constants_keys: List[str]):
        """

        Args:
            elements: objects that make up the ensemble, presumable all of the same type
            constants_keys: keys not average over, instead they will be taken from the first output
        """
        self.elements = elements
        self.constant_keys = constants_keys

    def __call__(self, f: Callable, *args, **kwargs):
        """

        Args:
            f:  a function to call. Probably of the form Class.method, because we will pass elements as "self"
            *args:
            **kwargs:

        Returns:

        """
        outputs = []
        for element in self.elements:
            output = f(element, *args, **kwargs)
            outputs.append(output)

        if isinstance(outputs[0], dict):
            outputs_dict = sequence_of_dicts_to_dict_of_tensors(outputs)

            nonconst_dict = {}
            # first just copy only the keys we want to take mean over
            for k, v in outputs_dict.items():
                if k not in self.constant_keys:
                    nonconst_dict[k] = v

            mean = {k: tf.math.reduce_mean(v, axis=0) for k, v in nonconst_dict.items()}
            stdev = {k: tf.math.reduce_std(v, axis=0) for k, v in nonconst_dict.items()}

            # then add back the keys we left out
            for k in self.constant_keys:
                # here is where we assume they're the same, and so we just take the first one
                mean[k] = outputs[0][k]
        elif isinstance(outputs[0], tf.Tensor) or isinstance(outputs[0], np.ndarray):
            mean = tf.math.reduce_mean(outputs, axis=0)
            stdev = tf.math.reduce_std(outputs, axis=0)
        else:
            raise NotImplementedError(f"Ensemble: Unimplemented return type {type(outputs[0])}")

        return mean, stdev
