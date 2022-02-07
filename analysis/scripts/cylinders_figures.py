import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from link_bot_pycommon.metric_utils import dict_to_pvalue_table


def main():
    df = pd.read_csv("/media/shared/cylinders_ablations.csv")

    method_name_map = {
        'aug':               'Augmentation (full method)',
        r'h50-60\+vel$':     'No Augmentation (baseline)',
        'no_occupancy':      'Augmentation (no occupancy)',
        'no_invariance':     'Augmentation (no transf. val.)',
        'no_delta_min_dist': 'Augmentation (no delta min dist)',
        'noise':             'Gaussian Noise (baseline)',
        'vae':             'VAE (baseline)',
    }

    palette = {
        'Augmentation (full method)':       '#d55e00',
        'No Augmentation (baseline)':       '#cc79a7',
        'Augmentation (no occupancy)':      '#0072b2',
        'Augmentation (no transf. val.)':   '#f0e442',
        'Augmentation (no delta min dist)': '#009e73',
        'Gaussian Noise (baseline)':        '#aaaaaa',
        'VAE (baseline)':                   '#eeee99',
    }

    for k, v in method_name_map.items():
        indices, = np.where(df['eval_dataset'].str.contains(k))
        df.loc[indices, 'method_name'] = v

    print_metrics(df, method_name_map, 'Mean Position Error')
    print_metrics(df, method_name_map, 'Mean Velocity Error')

    main_order = [
        'Augmentation (full method)',
        'No Augmentation (baseline)',
        'Gaussian Noise (baseline)',
    ]
    ablation_order = [
        'Augmentation (full method)',
        'Augmentation (no transf. val.)',
        'Augmentation (no delta min dist)',
        'Augmentation (no occupancy)',
        'No Augmentation (baseline)',
    ]

    plt.style.use('paper')
    sns.set(rc={'figure.figsize': (10, 4)})
    sns.set(font_scale=2)

    def barplot(x, order, title, v):
        plt.figure()
        ax = sns.barplot(data=df, order=order, y='method_name', x=x, ci=95,
                         palette=palette, orient='h', errwidth=10)
        ax.set_ylabel("")
        ax.set_title(title)
        plt.savefig(f"/media/shared/cylinders_{title.lower().replace(' ', '')}-{v}.png", dpi=200)

    barplot("Mean Position Error", main_order, 'Planar Pushing, Position Error', 'main')
    barplot("Mean Position Error", ablation_order, 'Planar Pushing, Position Error', 'ablations')
    barplot("Mean Velocity Error", main_order, 'Planar Pushing, Velocity Error', 'main')
    barplot("Mean Velocity Error", ablation_order, 'Planar Pushing, Velocity Error', 'ablations')

    plt.close()




if __name__ == '__main__':
    main()
