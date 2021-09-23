# importing the required packages
import pathlib
import time
from threading import Thread

import cv2
import numpy as np
import imageio
from mss import mss

import link_bot_pycommon.pycommon


def record(parent):
    images = []
    custom_duration = []

    long = 1.0
    short = 0.2

    region = {'top': parent.y, 'left': parent.x, 'width': parent.w, 'height': parent.h}
    color = (255, 255, 255)
    full_filename = parent.outdir / parent.filename

    with mss() as sct:
        while parent.recording:
            img = sct.grab(region)
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.putText(frame,
                                text=link_bot_pycommon.pycommon.as_posix(),
                                org=(5, 20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.4,
                                color=color,
                                thickness=1)
            images.append(frame)
            custom_duration.append(short)  # all frames are short
            time.sleep(short)

    # except the first and last frames, which are longer
    custom_duration[0] = long
    custom_duration[-1] = long

    imageio.mimsave(full_filename, images, duration=custom_duration, loop=0)


class ScreenRecorder:

    def __init__(self, outdir: pathlib.Path):
        self.filename = "plan_and_execution.gif"
        self.outdir = outdir
        self.x = 570
        self.y = 100
        self.w = 1300
        self.h = 800
        self.recording = False  # shared data!
        self.thread = Thread(target=record, args=(self,))

    def start(self):
        self.recording = True
        self.thread.start()

    def stop(self):
        self.recording = False
        self.thread.join()
