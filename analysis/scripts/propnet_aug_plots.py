import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from link_bot_pycommon.wandb_to_df import wandb_to_df


def main():
    np.set_printoptions(linewidth=300)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    df = wandb_to_df()
    m = 'method_name'
    map = {
        '/media/shared/fwd_model_data/h50-60+vel+aug-25-1638056556':    'Augmentation (full method)',
        '/media/shared/fwd_model_data/h50-60+vel':                      'No Augmentation (baseline)',
        '/media/shared/fwd_model_data/h50-60+vel+aug-25-simple-noise1': 'Gaussian Nose (baseline)',
    }
    df = df.loc[df['eval_dataset'].isin(map.keys())]
    df = df.loc[df['tags'].apply(lambda l: 'eval' in l and 'odd' not in l and 'bad' not in l)]
    df[m] = df['eval_dataset'].map(map)

    plt.style.use('paper')
    plt.rcParams['figure.figsize'] = (7, 4)
    sns.set(rc={'figure.figsize': (7, 4)})

    ax = sns.barplot(data=df,
                     y=m,
                     x='mean_error_pos',
                     hue=m,
                     palette='colorblind',
                     ci=95,
                     orient='h',
                     order=list(map.values()),
                     errwidth=5,
                     dodge=False)
    ax.set_ylabel("Method")
    ax.set_xlabel("Position Error (meters)")
    ax.set_title("Position Error")
    ax.ticklabel_format(axis='x', style='sci', scilimits=[-3,3])
    plt.savefig("propnet_aug_mean_error_pos.png")
    plt.show()


if __name__ == '__main__':
    main()
