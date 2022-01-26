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
             figsize=None,
             ci=100):
    fig = plt.figure(figsize=figsize)
    ax = sns.lineplot(
        data=df,
        x=x,
        y=metric,
        hue=hue,
        style=style,
        palette='colorblind',
        ci=ci,
        estimator='mean',
    )
    plt.plot([], [], ' ', label=f"shaded {ci}% c.i.")
    ax.set_title(title)
    return fig, ax


def violinplot(df, outdir, x: str, y: str, title: str, hue: Optional[str] = None, save: bool = True):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.violinplot(
        ax=ax,
        data=df,
        x=x,
        y=y,
        palette='colorblind',
        hue=hue,
        linewidth=4,
    )
    ax.set_title(title)
    if save:
        plt.savefig(outdir / f'{y}.png')
    return fig, ax


def barplot(df, outdir, x: str, y: str, title: str, hue: Optional[str] = None, ci=100):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        ax=ax,
        data=df,
        x=x,
        y=y,
        palette='colorblind',
        linewidth=5,
        ci=ci,
        hue=hue,
    )
    plt.plot([], [], ' ', label=f"shaded {ci}% c.i.")
    ax.set_title(title)
    plt.savefig(outdir / f'{x}-vs-{y}.png')
    return fig, ax
