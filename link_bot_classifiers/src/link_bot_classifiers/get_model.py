from link_bot_classifiers import nn_classifier, nn_recovery_model, nn_recovery_model2


def get_model(model_class_name):
    # NOTE: the confusing names are here for backwards compatibility with old pretrained models
    if model_class_name in ['nn_classifier2', 'nn_classifier']:
        return nn_classifier.NNClassifier
    elif model_class_name in ['recovery', 'nn']:
        return nn_recovery_model.NNRecoveryModel
    elif model_class_name in ['recovery2']:
        return nn_recovery_model2.NNRecoveryModel
    else:
        raise ValueError
