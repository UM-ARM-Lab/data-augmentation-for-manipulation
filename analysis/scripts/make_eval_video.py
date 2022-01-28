import argparse
import pathlib

from moviepy.editor import *

from link_bot_pycommon.args import int_set_arg


def hold_end(clip, duration):
    end = clip.subclip(clip.duration - 0.1, clip.duration)
    end.duration = duration
    return end


def eval_video(args):
    w = 1920
    speed = 10
    for root in args.roots:
        method_iteration_videos = []
        for iteration in args.iterations:

            iteration_dir = root / 'planning_results' / f"iteration_{iteration:04d}_planning"
            iteration_video_filenames = filter(lambda f: 'reset' not in f.as_posix(), iteration_dir.glob("*.avi"))

            method_iteration_clips = []
            for iteration_video_filename in iteration_video_filenames:
                method_iteration_clip = VideoFileClip(iteration_video_filename.as_posix(), audio=False)
                method_iteration_clip = method_iteration_clip.resize(width=w).speedx(speed)
                method_iteration_clips.append(method_iteration_clip)
            final_frame_pause = hold_end(method_iteration_clip, 3)
            method_iteration_video = concatenate_videoclips(method_iteration_clips + [final_frame_pause])

            iter_txt = TextClip(f'Iteration {iteration + 1}',
                                font='Ubuntu-Bold',
                                fontsize=45,
                                color='black',
                                stroke_color='white',
                                stroke_width=1.0,
                                method='caption',
                                size=method_iteration_video.size,
                                align='South')
            method_iteration_video_w_txt = CompositeVideoClip([method_iteration_video, iter_txt])
            method_iteration_video_w_txt.duration = method_iteration_video.duration

            method_iteration_videos.append(method_iteration_video_w_txt)

        outfilename = root / 'final.avi'
        method_concat = concatenate_videoclips(method_iteration_videos)

        speedup_txt = TextClip(f'{speed}x',
                               font='Ubuntu-Bold',
                               fontsize=40,
                               color='black',
                               stroke_color='white',
                               stroke_width=1.0)
        method_final = CompositeVideoClip([method_concat, speedup_txt])
        method_final.duration = method_concat.duration

        method_final.write_videofile(outfilename.as_posix(), fps=60, codec='libx264')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('roots', type=pathlib.Path, nargs='+')
    parser.add_argument('iterations', type=int_set_arg)

    args = parser.parse_args()

    # generate side-by-sides for each method at the given iterations

    eval_video(args)


if __name__ == '__main__':
    main()
