from typing import Optional

import seaborn as sns
from matplotlib import pyplot as plt


def try_set_violinplot_color(parts, key, color):
    if key in parts:
        parts[key].set_edgecolor(color)


def lineplot(df,
             x: str,
             metric: str,
             title: str,
             hue: Optional[str] = None,
             style: Optional[str] = None,
             figsize=None):
    fig = plt.figure(figsize=figsize)
    ax = sns.lineplot(
        data=df,
        x=x,
        y=metric,
        hue=hue,
        style=style,
        palette='colorblind',
        ci=100,
        estimator='mean',
    )
    ax.set_title(title)
    return fig, ax


def violinplot(df, outdir, x: str, y: str, title: str, hue: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.violinplot(
        ax=ax,
        data=df,
        x=x,
        y=y,
        hue=hue,
        linewidth=4,
    )
    ax.set_title(title)
    plt.savefig(outdir / f'{y}.png')
