import pathlib

from link_bot_classifiers import train_test_classifier


def eval_proxy_datasets(checkpoints, batch_size=128, **kwargs):
    dataset_dirs = [
        pathlib.Path("/media/shared/classifier_data/car_no_classifier_eval/"),
        pathlib.Path("/media/shared/classifier_data/car_heuristic_classifier_eval2/"),
        pathlib.Path("/media/shared/classifier_data/val_car_feasible_1614981888+op2/"),
    ]
    train_test_classifier.eval_n_main(dataset_dirs=dataset_dirs,
                                      checkpoints=checkpoints,
                                      batch_size=batch_size,
                                      mode='all',
                                      **kwargs)
