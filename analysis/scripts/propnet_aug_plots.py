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
        '/media/shared/fwd_model_data/h50-60+vel+aug-25-1638056556': 'Augmentation (full method)',
        '/media/shared/fwd_model_data/h50-60+vel':                   'No Augmentation (baseline)',
        # '/media/shared/fwd_model_data/h50-60+vel+aug-25-simple-noise1': 'Gaussian Noise (baseline)',
        # '/media/shared/fwd_model_data/h50-60+vel+aug-25-1643907565+vae': 'VAE (baseline)',
    }
    df = df.loc[df['dataset_dir'].isin(method_name_map.keys())]
    metric_name = 'mean_error_pos'
    df = df.loc[df[metric_name] != None]
    df = df.loc[df['scenario'] == 'cylinders']
    df = df.loc[df['tags'].apply(lambda l: 'eval' in l and 'odd' not in l and 'bad' not in l)]
    df[m] = df['eval_dataset'].map(method_name_map)
    df[metric_name] = df[metric_name].astype(np.float32)

    print_metrics(df, method_name_map, metric_name)

    plt.style.use('paper')
    plt.rcParams['figure.figsize'] = (8, 5)
    sns.set(rc={'figure.figsize': (8, 5)})

    ax = sns.violinplot(data=df,
                        y=m,
                        x=metric_name,
                        hue=m,
                        palette='colorblind',
                        ci=95,
                        orient='h',
                        order=list(method_name_map.values()),
                        linewidth=5,
                        dodge=False)
    ax.set_ylabel("Method")
    ax.set_xlabel("Position Error (meters)")
    ax.set_title("Position Error")
    ax.ticklabel_format(axis='x', style='sci', scilimits=[-3, 3])
    plt.savefig("propnet_aug_mean_error_pos.png")

    ax = sns.violinplot(data=df,
                        y=m,
                        x='penetration',
                        hue=m,
                        palette='colorblind',
                        ci=95,
                        orient='h',
                        order=list(method_name_map.values()),
                        linewidth=5,
                        dodge=False)
    ax.set_ylabel("Method")
    ax.set_xlabel("Penetration Error")
    ax.set_title("Penetration Error")
    ax.ticklabel_format(axis='x', style='sci', scilimits=[-3, 3])
    plt.savefig("propnet_aug_mean_error_penetration.png")
    plt.show()


def print_metrics(df, method_name_map, metric_name):
    print()
    print()
    mpe_dict = {}

    for n in np.unique(list(method_name_map.values())):
        if n == 'nan':
            continue
        metric_values = df.loc[df['method_name'] == n][metric_name].values
        mpe_dict[n] = metric_values
        print(f"{n:30s} {np.mean(metric_values):.4f} {np.std(metric_values):.4f}")
    print()
    print(dict_to_pvalue_table(mpe_dict))


if __name__ == '__main__':
    main()
