import json
import pathlib
import pickle
import shutil
from typing import Dict, Any, Optional

import hjson

from link_bot_pycommon.serialization import MyHJsonSerializer


def guess_mode_for_serializer(mode, serializer):
    if serializer in [hjson, json]:
        return mode
    if serializer in [pickle]:
        return mode + 'b'


def read_logfile(logfile_name: pathlib.Path, serializer=hjson):
    with logfile_name.open(guess_mode_for_serializer('r', serializer)) as logfile:
        log = serializer.load(logfile)
    return log


def write_logfile(log: Dict, logfile_name: pathlib.Path, serializer=MyHJsonSerializer):
    # prevents the user from losing the logfile by interrupting and killing the program
    temp_logfile_name = logfile_name.parent / (logfile_name.name + ".tmp")
    with temp_logfile_name.open(guess_mode_for_serializer('w', serializer)) as logfile:
        serializer.dump(log, logfile)
    shutil.copy(temp_logfile_name, logfile_name)


class JobChunker:

    def __init__(self,
                 logfile_name: pathlib.Path,
                 root_log: Optional[Dict] = None,
                 log: Optional[Dict] = None,
                 serializer=hjson):
        self.logfile_name = logfile_name
        self.serializer = serializer
        if root_log is not None:
            self.root_log = root_log
        else:
            if not logfile_name.exists():
                self.logfile_name.parent.mkdir(exist_ok=True, parents=True)
                self.root_log = {}
            else:
                self.root_log = read_logfile(self.logfile_name, serializer=self.serializer)
        if log is not None:
            self.log = log
        else:
            self.log = self.root_log

    def store_result(self, key: str, result: Any, save=True):
        self.log[key] = result
        if save:
            self.save()

    def store_results(self, update_dict: Dict, save=True):
        self.log.update(update_dict)
        if save:
            self.save()

    def save(self):
        write_logfile(self.root_log, self.logfile_name, serializer=self.serializer)

    def get_result(self, key: str):
        return self.log.get(key, None)

    def has_result(self, key: str):
        return key in self.log

    def setup_key(self, key: str):
        if key not in self.log:
            self.log[key] = {}
        self.save()

    def sub_chunker(self, key: str):
        self.setup_key(key)
        sub_chunker = JobChunker(self.logfile_name, root_log=self.root_log, log=self.log[key])
        return sub_chunker

    def get(self, key: str):
        return self.log[key]

    def done(self, done_key='done'):
        self.log[done_key] = True
        self.save()

    def is_done(self, done_key='done'):
        return done_key in self.log and self.log[done_key]

    def load_or_prompt(self, k):
        v = self.get_result(k)
        if v is None:
            v = input(f"{k}:")

        self.store_result(k, v)

        return v

    def load_or_default(self, key: str, default):
        if key in self.log:
            return self.log[key]
        self.store_result(key, default)
        return default
