import argparse
import pathlib

from moviepy.editor import *

from link_bot_pycommon.args import int_set_arg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=pathlib.Path)
    parser.add_argument('trials', type=int_set_arg)

    args = parser.parse_args()

    args.root

    ukulele = VideoFileClip(clip_path, audio=False).subclip(60 + 33, 60 + 50).crop(486, 180, 1196, 570)

    txt = TextClip('Audrey', font='Amiri-regular', fontsize=35)

    painting_txt = (CompositeVideoClip([painting, txt.set_pos((10, 180))])
                    .add_mask()
                    .set_duration(3)
                    .crossfadein(0.5)
                    .crossfadeout(0.5))

    w, h = ukulele.size

    piano = (VideoFileClip("../../videos/douceamb.mp4", audio=False).
             subclip(30, 50).
             resize((w / 3, h / 3)).  # one third of the total screen
             margin(6, color=(255, 255, 255)).  # white margin
             margin(bottom=20, right=20, opacity=0).  # transparent
             set_pos(('right', 'bottom')))

    final = CompositeVideoClip(video_eleme)
    final.subclip(0, 5).write_videofile("../../ukulele.avi", fps=24, codec='libx264')


if __name__ == '__main__':
    main()
