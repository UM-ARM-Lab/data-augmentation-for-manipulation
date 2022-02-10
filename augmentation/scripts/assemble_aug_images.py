import argparse
import pathlib

import PIL.Image
from matplotlib import pyplot as plt

from moonshine.filepath_tools import load_hjson

fill = 0.96
scale = 6


def extent_for_pos(w, h, row, col):
    y = row * h
    yc = y + h / 2
    x = col * w
    xc = x + w / 2
    return [xc - w * fill / 2,
            xc + w * fill / 2,
            yc - h * fill / 2,
            yc + h * fill / 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--crop')
    args = parser.parse_args()

    root = pathlib.Path("anims")
    out_info = load_hjson(root / args.name / 'out_info.txt')

    plt.style.use("paper")

    def _crop(img):
        if args.crop:
            w, h = img.size
            left, right, top, bottom = [int(s) for s in args.crop.split(",")]
            return img.crop([left, top, w - right, h - bottom])
        return img

    images = {}
    for row, filenames in enumerate(out_info.values()):
        original_filename = filenames['original']
        output_filenames = filenames['outputs']
        original_img = _crop(PIL.Image.open(original_filename))
        images[row] = {
            'original': original_img,
            'outputs':  [
            ]
        }
        for col, f in enumerate(output_filenames):
            output_img = _crop(PIL.Image.open(f))
            images[row]['outputs'].append(output_img)

    n_rows = len(images)
    img_w, img_h = original_img.size
    img_w /= scale
    img_h /= scale
    text_y = n_rows * img_h + 8

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.set_xlim(0, 4 * img_w)
    ax.set_ylim(0, text_y)
    ax.axis('off')
    ax.axvline(img_w, color='k', linewidth=img_w * 0.02)

    ax.text(0.5 * img_w, text_y, "Original",
            horizontalalignment='center', verticalalignment='center', fontsize=24)

    ax.text(2.5 * img_w, text_y, "Augmentations",
            horizontalalignment='center', verticalalignment='center', fontsize=24)

    for row, images_row in images.items():
        original_img = images_row['original']
        outputs = images_row['outputs']
        ax.imshow(original_img, extent=extent_for_pos(img_w, img_h, row, 0))
        for col, output_img in enumerate(outputs):
            ax.imshow(output_img, extent=extent_for_pos(img_w, img_h, row, col + 1))

    fig_filename = root / f"{args.name}_aug_examples.png"
    plt.savefig(fig_filename.as_posix(), dpi=200)
    plt.show()


if __name__ == '__main__':
    main()
