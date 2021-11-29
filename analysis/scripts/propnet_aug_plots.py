import argparse
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns

from link_bot_pycommon.wandb_to_df import NAME_KEY, wandb_to_df
from moonshine.filepath_tools import load_hjson


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filename', type=pathlib.Path)

    args = parser.parse_args()

    config = load_hjson(args.config_filename)

    plt.style.use("paper")

    df = wandb_to_df(config)
    ax = sns.barplot(data=df, x=NAME_KEY, y='val_loss', palette='colorblind', ci=100)
    ax.set_xlabel("Method")
    ax.set_ylabel("Position Error (meters)")
    ax.set_title("Position Error")
    plt.show()


if __name__ == '__main__':
    main()
