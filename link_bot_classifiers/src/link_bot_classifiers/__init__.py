from link_bot_classifiers import nn_classifier


def get_model(model_class_name):
    if model_class_name == 'rnn':
        return nn_classifier.NNClassifier
    else:
        raise ValueError
