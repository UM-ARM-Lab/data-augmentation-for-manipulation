import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from link_bot_pycommon.metric_utils import dict_to_pvalue_table
from link_bot_pycommon.wandb_to_df import wandb_to_df


def main():
    np.set_printoptions(linewidth=300)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    df = wandb_to_df()
    m = 'method_name'
    method_name_map = {
        '/media/shared/fwd_model_data/h50-60+vel+aug-25-1638056556':     'Augmentation (full method)',
        '/media/shared/fwd_model_data/h50-60+vel':                       'No Augmentation (baseline)',
        '/media/shared/fwd_model_data/h50-60+vel+aug-25-simple-noise1':  'Gaussian Nose (baseline)',
        '/media/shared/fwd_model_data/h50-60+vel+aug-25-1643907565+vae': 'VAE (baseline)',
    }
    df = df.loc[df['eval_dataset'].isin(method_name_map.keys())]
    df = df.loc[df['tags'].apply(lambda l: 'eval' in l and 'odd' not in l and 'bad' not in l)]
    df[m] = df['eval_dataset'].map(method_name_map)

    print_metrics(df, method_name_map, 'mean_error_pos')

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
                     order=list(method_name_map.values()),
                     errwidth=5,
                     dodge=False)
    ax.set_ylabel("Method")
    ax.set_xlabel("Position Error (meters)")
    ax.set_title("Position Error")
    ax.ticklabel_format(axis='x', style='sci', scilimits=[-3, 3])
    plt.savefig("propnet_aug_mean_error_pos.png")
    plt.show()


def print_metrics(df, method_name_map, metric_name):
    print()
    print()
    mpe_dict = {}

    for n in np.unique(list(method_name_map.values())):
        if n == 'nan':
            continue
        success_rates = df.loc[df['method_name'] == n][metric_name].values
        mpe_dict[n] = success_rates
        print(f"{n:30s} {np.mean(success_rates):.4f} {np.std(success_rates):.4f}")
    print()
    print(dict_to_pvalue_table(mpe_dict))


if __name__ == '__main__':
    main()
