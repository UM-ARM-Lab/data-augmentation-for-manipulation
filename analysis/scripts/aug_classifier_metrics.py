#!/usr/bin/env python
from arc_utilities import ros_init
from link_bot_data.dynamodb_utils import get_classifier_df
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


@ros_init.with_ros("analyse_planning_results")
def main():
    df = get_classifier_df()

    z = df.loc[
        df['dataset_dirs'].str.contains("/media/shared/ift/v6.3-0/classifier_datasets/iteration_00") &
        ~df['dataset_dirs'].str.contains(",") &
        df['classifier'].str.contains("99")
        ]

    z = z.copy()
    z['used_augmentation'] = ~z['classifier'].str.contains("baseline")
    m = ['accuracy', 'precision', 'accuracy on positives', 'accuracy on negatives', 'used_augmentation']
    h = z.groupby(["classifier"]).agg('mean')[m]
    q = h.groupby(["used_augmentation"]).agg('mean')

    print(q)


if __name__ == '__main__':
    main()
