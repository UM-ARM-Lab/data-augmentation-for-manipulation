import link_bot_classifiers.nn_recovery_model
from link_bot_classifiers import nn_classifier, nn_recovery_policy


def get_model(model_class_name):
    if model_class_name == 'rnn':
        return nn_classifier.NNClassifier
    if model_class_name == 'nn':
        return link_bot_classifiers.nn_recovery_model.NNRecoveryModel
    else:
        raise ValueError
