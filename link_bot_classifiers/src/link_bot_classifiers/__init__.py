from link_bot_classifiers import nn_classifier, nn_recovery_policy


def get_model(model_class_name):
    if model_class_name == 'rnn':
        return nn_classifier.NNClassifier
    if model_class_name == 'nn':
        return nn_recovery_policy.NNRecoveryModel
    else:
        raise ValueError
