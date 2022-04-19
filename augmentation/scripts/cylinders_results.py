import pathlib

import pandas as pd
from matplotlib import pyplot as plt

from analysis.results_figures import barplot


def main():
    plt.style.use("paper")

    results = pd.DataFrame([
        ["Augmentation (full method)", 0.00155],
        ["Augmentation (full method)", 0.0009148],
        ["Augmentation (full method)", 0.001089],
        ["Augmentation (full method)", 0.00113],
        ["Augmentation (full method)", 0.001308],
        ["Augmentation (full method)", 0.001321],
        ["Augmentation (full method)", 0.00143],
        ["Augmentation (full method)", 0.001443],
        ["Augmentation (full method)", 0.001501],
        ["Augmentation (full method)", 0.001581],
        ["Augmentation (full method)", 0.001582],
        ['No Augmentation (baseline)', 0.001238],
        ['No Augmentation (baseline)', 0.001254],
        ['No Augmentation (baseline)', 0.001343],
        ['No Augmentation (baseline)', 0.0014],
        ['No Augmentation (baseline)', 0.001415],
        ['No Augmentation (baseline)', 0.001565],
        ['No Augmentation (baseline)', 0.001625],
        ['No Augmentation (baseline)', 0.001656],
        ['No Augmentation (baseline)', 0.001825],
        ['No Augmentation (baseline)', 0.002042],
        ['VAE Augmentation (baseline)', 0.0021758],
        ['VAE Augmentation (baseline)', 0.001803],
        ['VAE Augmentation (baseline)', 0.002656],
        ['Gaussian Noise (baseline)', 0.001872],
        ['Gaussian Noise (baseline)', 0.002378],
        ['Gaussian Noise (baseline)', 0.003654],
        ['Gaussian Noise (baseline)', 0.003759],
    ],
        columns=['method_name', 'Position Error']
    )

    fig, ax = barplot(results, pathlib.Path("results"), y='method_name', x='Position Error', title='Position Error')
    ax.set_ylabel("")

    plt.show()


if __name__ == '__main__':
    main()
