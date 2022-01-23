import argparse
import pathlib

import PIL.Image
from matplotlib import pyplot as plt

from moonshine.filepath_tools import load_hjson

s = 100
w = 0.96


def extent_for_pos(row, col):
    y = row * s
    yc = y + s / 2
    x = col * s
    xc = x + s / 2
    return [xc - s * w / 2, xc + s * w / 2, yc - s * w / 2, yc + s * w / 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()

    root = pathlib.Path("anims")
    out_info = load_hjson(root / args.name / 'out_info.txt')

    plt.style.use("paper")

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.set_xlim(0, 4 * s)
    ax.set_ylim(0, 2.05 * s)
    ax.axis('off')
    ax.axvline(s, color='k', linewidth=s * 0.05)

    ax.text(0.5 * s, 2.05 * s, "Original", horizontalalignment='center', verticalalignment='center')

    ax.text(2.5 * s, 2.05 * s, "Augmentations", horizontalalignment='center', verticalalignment='center')

    for row, filenames in enumerate(out_info.values()):
        original_filename = filenames['original']
        output_filenames = filenames['outputs']
        original_img = PIL.Image.open(original_filename)
        ax.imshow(original_img, extent=extent_for_pos(row, 0))
        for col, f in enumerate(output_filenames):
            output_img = PIL.Image.open(f)
            ax.imshow(output_img, extent=extent_for_pos(row, col + 1))

    fig_filename = root / f"{args.name}_aug_examples.png"
    plt.savefig(fig_filename.as_posix(), dpi=200)
    plt.show()


if __name__ == '__main__':
    main()
