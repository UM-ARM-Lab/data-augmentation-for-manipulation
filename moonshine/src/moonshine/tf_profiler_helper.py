import sys
from typing import Tuple

import tensorflow as tf
from colorama import Back, Fore


class TFProfilerHelper:

    def __init__(self, profile_arg: Tuple[int], train_logdir: str):
        self.train_logdir = train_logdir
        if profile_arg is None:
            self.start_batch = sys.maxsize
            self.stop_batch = -1
        elif isinstance(profile_arg, tuple):
            self.start_batch = profile_arg[0]
            self.stop_batch = profile_arg[1]
        else:
            raise NotImplementedError()
        self.started = False
        self.finished = False

    def start(self, batch_idx: int, epoch: int):
        if batch_idx >= self.start_batch and not self.started and not self.finished and epoch == 1:
            self.started = True
            print(Back.WHITE + Fore.BLACK + "Starting Profiler" + Fore.RESET + Back.RESET)
            options = tf.profiler.experimental.ProfilerOptions(python_tracer_level=1)
            tf.profiler.experimental.start(self.train_logdir, options)
        return TFProfilerStopper(batch_idx, epoch, self)

    def stop_internal(self):
        """ Do not call this directly, use the TFProfilerStopper """
        self.started = False
        self.finished = True
        print(Back.WHITE + Fore.BLACK + "Stopping Profiler" + Fore.RESET + Back.RESET)
        tf.profiler.experimental.stop()

    def __del__(self):
        if self.started and not self.finished:
            print(Fore.RED + "Stopping profiler upon destruction!" + Fore.RESET)
            self.stop_internal()
            tf.profiler.experimental.stop()


class TFProfilerStopper:

    def __init__(self, batch_idx, epoch, parent: TFProfilerHelper):
        self.parent = parent
        self.batch_idx = batch_idx
        self.epoch = epoch

    def stop(self):
        if not self.parent.started or self.parent.finished or self.epoch > 1:
            return
        if self.batch_idx >= self.parent.stop_batch:
            self.parent.stop_internal()
