import pathlib
from typing import Dict, List, Optional

import tensorflow as tf

from link_bot_classifiers.uncertainty import make_max_class_prob
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.ensemble import Ensemble2
from moonshine.filepath_tools import load_params


class BaseConstraintChecker:

    def __init__(self, path: pathlib.Path, scenario: ExperimentScenario):
        self.path = path
        self.scenario = scenario
        self.horizon = 2
        self.hparams = load_params(self.path.parent)
        self.name = self.__class__.__name__

    def check_constraint_from_example(self, example: Dict, training: Optional[bool] = False):
        raise NotImplementedError()

    def check_constraint_tf_batched(self,
                                    environment: Dict,
                                    states: Dict,
                                    actions: Dict,
                                    batch_size: int,
                                    state_sequence_length: int):
        raise NotImplementedError()

    def check_constraint_tf(self,
                            environment: Dict,
                            states_sequence: List[Dict],
                            actions: List[Dict]):
        raise NotImplementedError()

    def check_constraint(self,
                         environment: Dict,
                         states_sequence: List[Dict],
                         actions: List[Dict]):
        c = self.check_constraint_tf(environment, states_sequence, actions)
        return c.numpy()


def classifier_ensemble_check_constraint(A, B, mean, stdev):
    x = tf.stack([stdev, make_max_class_prob(mean)], axis=1)
    y = tf.matmul(x, tf.transpose(A)) + B
    y = tf.squeeze(y, axis=1)
    # since the calling code will use it's own threshold for accept in the interface (0, 1)
    # we use sigmoid to map from (-inf,inf) to (0, 1)
    too_uncertain = tf.keras.activations.sigmoid(y)

    p_accept = tf.minimum(too_uncertain, mean)
    return p_accept


class ConstraintCheckerEnsemble(BaseConstraintChecker):
    def __init__(self, path, elements, constants_keys: List[str]):
        self.ensemble = Ensemble2(elements, constants_keys)
        m0 = self.ensemble.elements[0]
        self.element_class = m0.__class__

        BaseConstraintChecker.__init__(self, path, m0.scenario)

        self.A = tf.constant(self.hparams['A'], tf.float32)
        self.B = tf.constant(self.hparams['B'], tf.float32)

    def check_constraint_tf(self, *args, **kwargs):
        mean, stdev = self.ensemble(self.element_class.check_constraint_tf, *args, **kwargs)
        constraint_satisfied = classifier_ensemble_check_constraint(self.A, self.B, mean, stdev)

        # DEBUGGING
        self.scenario.plot_stdev(stdev)
        # END DEBUGGING

        return constraint_satisfied

    def check_constraint_tf_batched(self, *args, **kwargs):
        mean, stdev = self.ensemble(self.element_class.check_constraint_tf_batched, *args, **kwargs)
        return mean, stdev

    def check_constraint_from_example(self, *args, **kwargs):
        mean, stdev = self.ensemble(self.element_class.check_constraint_from_example, *args, **kwargs)
        return mean, stdev
