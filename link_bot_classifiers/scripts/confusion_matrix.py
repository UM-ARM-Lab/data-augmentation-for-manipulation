#!/usr/bin/env python
import argparse
import logging
import pathlib

import pandas as pd
import tensorflow as tf

from arc_utilities import ros_init
from link_bot_classifiers.train_test_classifier import eval_main
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_sequences as seq2dict


@ros_init.with_ros("confusion_matrix")
def main():
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("datasets_dir", type=pathlib.Path)
    parser.add_argument("models_dir", type=pathlib.Path)
    parser.add_argument("--take", type=int)

    args = parser.parse_args()

    dataset_dirs = list(args.datasets_dir.iterdir())
    model_dirs = list(args.models_dir.iterdir())

    mode = 'train'
    outdir = pathlib.Path('results') / args.datasets_dir.name / mode
    outdir.mkdir(exist_ok=True, parents=True)

    batch_size = 32

    metrics_for_datasets = []
    for dataset_dir in dataset_dirs:
        metrics_for_dataset = []
        for model_dir in model_dirs:
            print(dataset_dir.name, model_dir.name)
            metrics_for_dataset_and_model = eval_main(dataset_dirs=[dataset_dir],
                                                      mode=mode,
                                                      batch_size=batch_size,
                                                      use_gt_rope=True,
                                                      threshold=None,
                                                      take=args.take * batch_size,
                                                      checkpoint=model_dir / 'best_checkpoint')
            m = {k: v.result().numpy().squeeze() for k, v in metrics_for_dataset_and_model.items()}
            metrics_for_dataset.append(m)

        metrics_for_datasets.append(metrics_for_dataset)

    metrics_dict = seq2dict([seq2dict(m) for m in metrics_for_datasets])

    headers = [d.name for d in dataset_dirs]
    for metric_name, confusion_matrix in metrics_dict.items():
        df = pd.DataFrame(confusion_matrix)
        df.insert(0, '', headers)
        print(metric_name)
        print(df)

        metric_name = metric_name.replace("/", "-").replace(" ", "_")
        outfilename = outdir / (metric_name + ".csv")
        df.to_csv(outfilename, sep='\t')


if __name__ == '__main__':
    main()
