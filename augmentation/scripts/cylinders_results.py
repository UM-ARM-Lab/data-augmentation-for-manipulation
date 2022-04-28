import pathlib

import pandas as pd
from matplotlib import pyplot as plt

from analysis.results_figures import barplot
from link_bot_pycommon.metric_utils import df_to_pvalue_table


def main():
    plt.style.use("paper")

    results = pd.DataFrame([
        ["Augmentation (full method)", 0.000915],
        ["Augmentation (full method)", 0.001089],
        ["Augmentation (full method)", 0.001130],
        ["Augmentation (full method)", 0.001308],
        ["Augmentation (full method)", 0.001321],
        ["Augmentation (full method)", 0.001430],
        ["Augmentation (full method)", 0.001443],
        ["Augmentation (full method)", 0.001501],
        ["Augmentation (full method)", 0.001581],
        ["Augmentation (full method)", 0.001582],
        ['No Augmentation (baseline)', 0.001238],
        ['No Augmentation (baseline)', 0.001254],
        ['No Augmentation (baseline)', 0.001343],
        ['No Augmentation (baseline)', 0.001400],
        ['No Augmentation (baseline)', 0.001415],
        ['No Augmentation (baseline)', 0.001565],
        ['No Augmentation (baseline)', 0.001625],
        ['No Augmentation (baseline)', 0.001656],
        ['No Augmentation (baseline)', 0.001825],
        ['No Augmentation (baseline)', 0.002042],
        ['VAE Augmentation (baseline)', 0.001803],
        ['VAE Augmentation (baseline)', 0.0021758],
        ['VAE Augmentation (baseline)', 0.002656],
        ['Gaussian Noise (baseline)', 0.001872],
        ['Gaussian Noise (baseline)', 0.002378],
        ['Gaussian Noise (baseline)', 0.003654],
        ['Gaussian Noise (baseline)', 0.003759],
    ],
        columns=['method_name', 'Position Error']
    )

    print()
    print("P-values:")
    print(df_to_pvalue_table(results, 'Position Error'))

    print()
    print("stds:")
    print(results.groupby("method_name").std())

    print()
    print("Means:")
    print(results.groupby("method_name").mean())

    fig, ax = barplot(results, pathlib.Path("results"), y='method_name', x='Position Error', title='Position Error',
                      figsize=(14, 6))
    ax.set_ylabel("")
    plt.savefig("results/cylinders_results1.png", dpi=300)


if __name__ == '__main__':
    main()
